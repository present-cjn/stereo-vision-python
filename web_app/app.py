import sys
import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

# --- 关键的路径设置 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
# --------------------

# --- 导入你自己的核心算法模块 ---
import config
from utils import file_utils, image_utils
from processing.stereo_matcher import StereoMatcher
from processing.reconstructor import Reconstructor
from visualization import visualizer

# --------------------------------

# 初始化 Flask 应用
# --- 核心修正：告诉 Flask 正确的静态文件夹路径 ---
app = Flask(__name__, static_folder='static')
# 我们将上传的文件和生成的结果都放在 static 文件夹下，方便Web访问
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'web_app', 'static', 'uploads')
RESULT_FOLDER = os.path.join(PROJECT_ROOT, 'web_app', 'static', 'results')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# 确保文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


@app.route('/')
def index():
    """渲染主页面 (index.html)"""
    return render_template('index.html')


@app.route('/run_stereo', methods=['POST'])
def run_stereo_endpoint():
    """处理立体匹配请求的 API 端点"""
    try:
        # --- 1. 处理文件上传 ---
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

        # --- 2. 调用你的核心双目视觉流水线 ---

        # 加载标定参数
        stereo_params = file_utils.load_stereo_params(config.CAMERA_PARAMS_PATH)
        if stereo_params is None:
            return jsonify({'error': '相机标定文件未找到!'}), 500

        # 读取上传的图像
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        if left_img is None or right_img is None:
            return jsonify({'error': '无法读取上传的图像文件'}), 400

        # --- 调用3参数版本的 rectify_stereo_pair ---
        left_rectified, right_rectified, Q = image_utils.rectify_stereo_pair(left_img, right_img, stereo_params)

        # 计算视差
        matcher = StereoMatcher()
        disparity_map = matcher.compute_disparity(left_rectified, right_rectified)

        # --- 3. 生成结果图像 ---
        disp_to_show = cv2.normalize(
            (disparity_map.astype(np.float32) / 16.0 - config.SGBM_MIN_DISPARITY) / config.SGBM_NUM_DISPARITIES, None,
            alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        colormap_disp = cv2.applyColorMap(disp_to_show, cv2.COLORMAP_JET)

        result_filename = f"result_{left_filename}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_path, colormap_disp)

        # --- 4. 返回结果图像的URL ---
        result_image_url = url_for('static', filename=f'results/{result_filename}')
        return jsonify({'result_image': result_image_url})

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)

