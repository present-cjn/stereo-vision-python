import sys
import os
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
from werkzeug.utils import secure_filename
from .tasks import run_stereo_reconstruction_task, run_calibration_task
from celery.result import AsyncResult
from .celery_app import celery_app  # 导入celery_app实例
import uuid

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

# --- 确保标定图片目录也存在 ---
CALIB_UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'data', 'calibration_images')
os.makedirs(CALIB_UPLOAD_FOLDER, exist_ok=True)



@app.route('/')
def index():
    # 将渲染的模板从 index.html 改为 workbench.html
    return render_template('workbench.html')


# 2. 用下面的函数替换掉旧的 /run_stereo
@app.route('/upload_images', methods=['POST'])
def upload_images_endpoint():
    """
    接收用户上传的左图、右图和标定文件，
    为本次任务创建一个唯一的文件夹，并保存文件。
    返回图片的公共URL和任务ID，用于后续步骤。
    """
    try:
        # 检查文件是否都已上传
        if 'left_image' not in request.files or 'right_image' not in request.files or 'params_file' not in request.files:
            return jsonify({'error': '文件上传不完整：需要左图像、右图像和标定文件'}), 400

        left_file = request.files['left_image']
        right_file = request.files['right_image']
        params_file = request.files['params_file']

        if left_file.filename == '' or right_file.filename == '' or params_file.filename == '':
            return jsonify({'error': '有文件未选择'}), 400

        # 为这次任务创建一个唯一的ID和文件夹
        task_id = str(uuid.uuid4())
        task_folder = os.path.join(app.config['UPLOAD_FOLDER'], task_id)
        os.makedirs(task_folder, exist_ok=True)

        # 保存文件
        left_filename = secure_filename(left_file.filename)
        right_filename = secure_filename(right_file.filename)
        params_filename = secure_filename(params_file.filename)

        left_path = os.path.join(task_folder, left_filename)
        right_path = os.path.join(task_folder, right_filename)
        params_path = os.path.join(task_folder, params_filename)

        left_file.save(left_path)
        right_file.save(right_path)
        params_file.save(params_path)

        # 构建前端可以访问的公共URL
        left_image_url = url_for('static', filename=f'uploads/{task_id}/{left_filename}')
        right_image_url = url_for('static', filename=f'uploads/{task_id}/{right_filename}')

        # 返回成功信息
        return jsonify({
            'message': '文件上传成功!',
            'task_id': task_id,
            'left_image_url': left_image_url,
            'right_image_url': right_image_url
        })

    except Exception as e:
        return jsonify({'error': f'服务器内部错误: {str(e)}'}), 500


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


@app.route('/start_calibration', methods=['POST'])
def start_calibration_endpoint():
    """接收标定参数和多张图片，启动后台标定任务。"""
    try:
        # 1. 获取表单数据
        corners = request.form.get('corners')
        size = request.form.get('size')
        if not corners or not size:
            return jsonify({'error': '标定板参数缺失'}), 400

        # 2. 处理多文件上传
        files = request.files.getlist('calib_images')
        if not files:
            return jsonify({'error': '未上传标定图片'}), 400

        # 清空旧的标定图片以避免混淆
        for f in os.listdir(CALIB_UPLOAD_FOLDER):
            os.remove(os.path.join(CALIB_UPLOAD_FOLDER, f))

        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                file.save(os.path.join(CALIB_UPLOAD_FOLDER, filename))

        # 3. 启动后台任务
        task = run_calibration_task.delay(CALIB_UPLOAD_FOLDER, corners, size)

        return jsonify({
            'task_id': task.id,
            'status_url': url_for('get_task_status', task_id=task.id)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download_params')
def download_params_endpoint():
    """提供标定结果文件的下载。"""
    try:
        return send_from_directory(
            os.path.join(PROJECT_ROOT, 'output'),
            'stereo_params.yml',
            as_attachment=True
        )
    except FileNotFoundError:
        return "标定文件未找到！", 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)