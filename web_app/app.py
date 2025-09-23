# web_app/app.py

import sys
import os
import uuid
import cv2
import numpy as np
import time
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

# --- 关键的路径设置 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# --- 导入你自己的核心算法模块 ---
import config
from utils import file_utils, image_utils
from processing.stereo_matcher import StereoMatcher

# --- Flask 应用初始化 ---
app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'web_app', 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('workbench.html')

@app.route('/upload_data', methods=['POST'])
def upload_data_endpoint():
    try:
        # ... (此处省略与之前版本相同的、正确的代码) ...
        if 'left_image' not in request.files or 'right_image' not in request.files or 'params_file' not in request.files: return jsonify({'error': '文件上传不完整'}), 400
        left_file, right_file, params_file = request.files['left_image'], request.files['right_image'], request.files['params_file']
        if not all([left_file.filename, right_file.filename, params_file.filename]): return jsonify({'error': '有文件未选择'}), 400
        task_id = str(uuid.uuid4())
        task_folder = os.path.join(app.config['UPLOAD_FOLDER'], task_id)
        os.makedirs(task_folder, exist_ok=True)
        left_filename, right_filename, params_filename = secure_filename(left_file.filename), secure_filename(right_file.filename), secure_filename(params_file.filename)
        left_path, right_path, params_path = os.path.join(task_folder, left_filename), os.path.join(task_folder, right_filename), os.path.join(task_folder, params_filename)
        left_file.save(left_path); right_file.save(right_path); params_file.save(params_path)
        left_image_url, right_image_url = url_for('static', filename=f'uploads/{task_id}/{left_filename}'), url_for('static', filename=f'uploads/{task_id}/{right_filename}')
        return jsonify({'message': '文件上传成功!', 'task_id': task_id, 'left_image_url': left_image_url, 'right_image_url': right_image_url})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': f'服务器内部错误: {str(e)}'}), 500

@app.route('/rectify_images', methods=['POST'])
def rectify_images_endpoint():
    try:
        # ... (此处省略与之前版本相同的、正确的代码) ...
        data = request.get_json(); task_id = data.get('task_id')
        if not task_id: return jsonify({'error': '缺少任务ID'}), 400
        task_folder = os.path.join(app.config['UPLOAD_FOLDER'], task_id)
        left_path = next((os.path.join(task_folder, f) for f in os.listdir(task_folder) if 'left' in f.lower() and f.endswith(('.png', '.jpg', '.jpeg'))), None)
        right_path = next((os.path.join(task_folder, f) for f in os.listdir(task_folder) if 'right' in f.lower() and f.endswith(('.png', '.jpg', '.jpeg'))), None)
        params_path = next((os.path.join(task_folder, f) for f in os.listdir(task_folder) if f.lower().endswith(('.yml', '.yaml'))), None)
        if not all([left_path, right_path, params_path]): return jsonify({'error': '任务文件夹中文件不完整或命名不规范'}), 400
        left_img, right_img, stereo_params = cv2.imread(left_path), cv2.imread(right_path), file_utils.load_stereo_params(params_path)
        left_rectified, right_rectified, Q = image_utils.rectify_stereo_pair(left_img, right_img, stereo_params)
        left_rect_filename, right_rect_filename = "left_rectified.png", "right_rectified.png"
        left_rect_path, right_rect_path = os.path.join(task_folder, left_rect_filename), os.path.join(task_folder, right_rect_filename)
        q_matrix_path = os.path.join(task_folder, "Q.npy")
        cv2.imwrite(left_rect_path, left_rectified); cv2.imwrite(right_rect_path, right_rectified); np.save(q_matrix_path, Q)
        left_rect_url = url_for('static', filename=f'uploads/{task_id}/{left_rect_filename}'); right_rect_url = url_for('static', filename=f'uploads/{task_id}/{right_rect_filename}')
        return jsonify({'message': '立体校正成功!', 'left_rectified_url': left_rect_url, 'right_rectified_url': right_rect_url})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': f'校正过程中发生错误: {str(e)}'}), 500

@app.route('/compute_disparity', methods=['POST'])
def compute_disparity_endpoint():
    try:
        data = request.get_json()
        task_id, params = data.get('task_id'), data.get('params')
        if not task_id or not params: return jsonify({'error': '缺少任务ID或SGBM参数'}), 400
        task_folder = os.path.join(app.config['UPLOAD_FOLDER'], task_id)
        left_rect_path, right_rect_path = os.path.join(task_folder, "left_rectified.png"), os.path.join(task_folder, "right_rectified.png")
        if not os.path.exists(left_rect_path) or not os.path.exists(right_rect_path): return jsonify({'error': '校正后的图像未找到'}), 400
        left_rectified, right_rectified = cv2.imread(left_rect_path), cv2.imread(right_rect_path)
        matcher = StereoMatcher(**params)
        disparity_map = matcher.compute_disparity(left_rectified, right_rectified)
        disp_raw_path = os.path.join(task_folder, "disparity_raw.npy"); np.save(disp_raw_path, disparity_map)
        disp_to_show = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        colormap_disp = cv2.applyColorMap(disp_to_show, cv2.COLORMAP_JET)
        timestamp = int(time.time() * 1000); disp_filename = f"disparity_{timestamp}.png"
        disp_path = os.path.join(task_folder, disp_filename); cv2.imwrite(disp_path, colormap_disp)
        disparity_url = url_for('static', filename=f'uploads/{task_id}/{disp_filename}')
        return jsonify({'message': '视差图计算成功!', 'disparity_map_url': disparity_url})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': f'计算视差时发生错误: {str(e)}'}), 500


@app.route('/reconstruct_3d', methods=['POST'])
def reconstruct_3d_endpoint():
    """
    阶段四：接收任务ID，执行三维重建。
    返回点云文件的下载链接和结果信息。
    """
    try:
        data = request.get_json()
        task_id = data.get('task_id')
        if not task_id:
            return jsonify({'error': '缺少任务ID'}), 400

        task_folder = os.path.join(app.config['UPLOAD_FOLDER'], task_id)

        # --- 1. 加载所需文件 ---
        disp_raw_path = os.path.join(task_folder, "disparity_raw.npy")
        left_rect_path = os.path.join(task_folder, "left_rectified.png")
        q_matrix_path = os.path.join(task_folder, "Q.npy")

        if not all(os.path.exists(p) for p in [disp_raw_path, left_rect_path, q_matrix_path]):
            return jsonify({'error': '重建所需文件不完整，请先完成前面步骤'}), 400

        disparity_map = np.load(disp_raw_path)
        left_rectified = cv2.imread(left_rect_path)
        Q = np.load(q_matrix_path)

        # --- 2. 执行重建 ---
        reconstructor = Reconstructor()
        # 注意：这里我们只用过滤后的点云来保存
        _, (points_filtered, colors_filtered) = reconstructor.reconstruct(disparity_map, left_rectified, Q)

        # --- 3. 保存点云文件 ---
        point_cloud_filename = f"point_cloud_{task_id}.ply"
        point_cloud_path = os.path.join(task_folder, point_cloud_filename)

        # 在保存前进行降采样
        factor = config.POINT_CLOUD_DOWNSAMPLE_FACTOR
        if factor > 1:
            points_to_save = points_filtered[::factor]
            colors_to_save = colors_filtered[::factor]
        else:
            points_to_save = points_filtered
            colors_to_save = colors_filtered

        file_utils.save_point_cloud(point_cloud_path, points_to_save, colors_to_save)

        # --- 4. 返回结果信息 ---
        point_cloud_url = url_for('static', filename=f'uploads/{task_id}/{point_cloud_filename}')

        return jsonify({
            'message': '三维重建成功!',
            'point_count': len(points_to_save),
            'point_cloud_url': point_cloud_url
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'重建时发生错误: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)