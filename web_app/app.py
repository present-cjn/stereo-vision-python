import sys
import os
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
from werkzeug.utils import secure_filename
from .tasks import run_stereo_reconstruction_task, run_calibration_task
from celery.result import AsyncResult
from .celery_app import celery_app  # 导入celery_app实例
import uuid
import numpy as np
import cv2

# --- 关键的路径设置 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# --------------------
import config
from utils import file_utils, image_utils
from processing.stereo_matcher import StereoMatcher
from processing.reconstructor import Reconstructor


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


# 2. 添加这个全新的函数
@app.route('/rectify_images', methods=['POST'])
def rectify_images_endpoint():
    """
    接收任务ID，执行立体校正，并返回校正后图像的URL。
    """
    try:
        data = request.get_json()
        task_id = data.get('task_id')
        if not task_id:
            return jsonify({'error': '缺少任务ID'}), 400

        task_folder = os.path.join(app.config['UPLOAD_FOLDER'], task_id)

        # --- 1. 定位文件路径 ---
        # (一个更健壮的设计会在这里查找文件名，但为简化，我们假设文件名已知)
        left_path = next(
            (os.path.join(task_folder, f) for f in os.listdir(task_folder) if f.lower().startswith('left')), None)
        right_path = next(
            (os.path.join(task_folder, f) for f in os.listdir(task_folder) if f.lower().startswith('right')), None)
        params_path = next(
            (os.path.join(task_folder, f) for f in os.listdir(task_folder) if f.lower().endswith(('.yml', '.yaml'))),
            None)

        if not all([left_path, right_path, params_path]):
            return jsonify({'error': '任务文件夹中文件不完整'}), 400

        # --- 2. 加载数据并执行校正 ---
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        stereo_params = file_utils.load_stereo_params(params_path)

        left_rectified, right_rectified, Q = image_utils.rectify_stereo_pair(left_img, right_img, stereo_params)

        # --- 3. 保存结果 ---
        left_rect_filename = f"left_rectified.png"
        right_rect_filename = f"right_rectified.png"
        left_rect_path = os.path.join(task_folder, left_rect_filename)
        right_rect_path = os.path.join(task_folder, right_rect_filename)
        q_matrix_path = os.path.join(task_folder, "Q.npy")

        cv2.imwrite(left_rect_path, left_rectified)
        cv2.imwrite(right_rect_path, right_rectified)
        np.save(q_matrix_path, Q)  # 将Q矩阵保存为numpy二进制文件

        # --- 4. 返回新图片的URL ---
        left_rect_url = url_for('static', filename=f'uploads/{task_id}/{left_rect_filename}')
        right_rect_url = url_for('static', filename=f'uploads/{task_id}/{right_rect_filename}')

        return jsonify({
            'message': '立体校正成功!',
            'left_rectified_url': left_rect_url,
            'right_rectified_url': right_rect_url
        })

    except Exception as e:
        return jsonify({'error': f'校正过程中发生错误: {str(e)}'}), 500


@app.route('/compute_disparity', methods=['POST'])
def compute_disparity_endpoint():
    """
    阶段三：接收任务ID和SGBM参数，计算视差图并返回其URL。
    这是一个同步、快速响应的接口，用于实时调参。
    """
    try:
        data = request.get_json()
        task_id = data.get('task_id')
        params = data.get('params')
        if not task_id or not params:
            return jsonify({'error': '缺少任务ID或SGBM参数'}), 400

        task_folder = os.path.join(app.config['UPLOAD_FOLDER'], task_id)

        # --- 1. 加载校正后的图像 ---
        left_rect_path = os.path.join(task_folder, "left_rectified.png")
        right_rect_path = os.path.join(task_folder, "right_rectified.png")

        if not os.path.exists(left_rect_path) or not os.path.exists(right_rect_path):
            return jsonify({'error': '校正后的图像未找到，请先执行校正步骤'}), 400

        left_rectified = cv2.imread(left_rect_path)
        right_rectified = cv2.imread(right_rect_path)

        # --- 2. 使用前端传来的参数实例化匹配器 ---
        # 我们使用 .get() 方法，如果前端没有提供某个参数，就从config中获取默认值
        matcher = StereoMatcher(
            minDisparity=params.get('minDisparity', config.SGBM_MIN_DISPARITY),
            numDisparities=params.get('numDisparities', config.SGBM_NUM_DISPARITIES),
            blockSize=params.get('blockSize', config.SGBM_BLOCK_SIZE),
            uniquenessRatio=params.get('uniquenessRatio', config.SGBM_UNIQUENESS_RATIO),
            speckleWindowSize=params.get('speckleWindowSize', config.SGBM_SPECKLE_WINDOW_SIZE),
            speckleRange=params.get('speckleRange', config.SGBM_SPECKLE_RANGE),
            disp12MaxDiff=params.get('disp12MaxDiff', config.SGBM_DISP12_MAX_DIFF)
            # 如果你在前端添加了 P1, P2 的滑块，也可以在这里用同样的方式获取
        )

        # --- 3. 计算并保存视差图 ---
        disparity_map = matcher.compute_disparity(left_rectified, right_rectified)

        # a) 将原始的、数据更丰富的16位视差图保存起来，供后续的三维重建使用
        disp_raw_path = os.path.join(task_folder, "disparity_raw.npy")
        np.save(disp_raw_path, disparity_map)

        # b) 创建用于前端显示的可视化版本 (8位彩色图)
        disp_to_show = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        colormap_disp = cv2.applyColorMap(disp_to_show, cv2.COLORMAP_JET)

        # 每次调参都生成一个带时间戳的图片，避免浏览器缓存旧的图片
        timestamp = int(time.time() * 1000)
        disp_filename = f"disparity_{timestamp}.png"
        disp_path = os.path.join(task_folder, disp_filename)
        cv2.imwrite(disp_path, colormap_disp)

        # --- 4. 返回新视差图的URL ---
        disparity_url = url_for('static', filename=f'uploads/{task_id}/{disp_filename}')

        return jsonify({
            'message': '视差图计算成功!',
            'disparity_map_url': disparity_url
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'计算视差时发生错误: {str(e)}'}), 500


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