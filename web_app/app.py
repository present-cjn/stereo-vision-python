import sys
import os
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from .tasks import run_stereo_reconstruction_task
from celery.result import AsyncResult
from .celery_app import celery_app  # 导入celery_app实例

# --- 关键的路径设置 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# --------------------

# 初始化 Flask 应用
app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'web_app', 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, 'web_app', 'static', 'results'), exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/run_stereo', methods=['POST'])
def run_stereo_endpoint():
    """
    接收图像上传，并启动一个后台 Celery 任务。
    立即返回一个任务 ID。
    """
    try:
        if 'left_image' not in request.files or 'right_image' not in request.files:
            return jsonify({'error': '左右图像文件缺失'}), 400

        left_file = request.files['left_image']
        right_file = request.files['right_image']
        left_filename = secure_filename(left_file.filename)
        right_filename = secure_filename(right_file.filename)
        left_path = os.path.join(app.config['UPLOAD_FOLDER'], left_filename)
        right_path = os.path.join(app.config['UPLOAD_FOLDER'], right_filename)
        left_file.save(left_path)
        right_file.save(right_path)

        # 核心修改：启动后台任务，而不是自己计算
        task = run_stereo_reconstruction_task.delay(left_path, right_path)

        # 返回任务ID和状态查询URL
        return jsonify({
            'task_id': task.id,
            'status_url': url_for('get_task_status', task_id=task.id)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/status/<task_id>')
def get_task_status(task_id):
    """
    前端轮询此端点来查询任务状态。
    """
    task_result = AsyncResult(task_id, app=celery_app)

    if task_result.state == 'PENDING':
        response = {'state': 'PENDING', 'status': '任务正在等待...', 'progress': 0}
    elif task_result.state == 'PROGRESS':
        response = {'state': 'PROGRESS', 'status': task_result.info.get('status', ''),
                    'progress': task_result.info.get('progress', 0)}
    elif task_result.state == 'SUCCESS':
        response = {'state': 'SUCCESS', 'status': '任务成功', 'progress': 100, 'result': task_result.result}
    else:  # FAILURE
        response = {'state': 'FAILURE', 'status': str(task_result.info), 'progress': 0}

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, port=5000)