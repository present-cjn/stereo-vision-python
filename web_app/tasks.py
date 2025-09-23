import sys
import os
import cv2
from .celery_app import celery_app
import numpy as np

# --- 关键的路径设置，确保 Celery worker 能找到核心模块 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# ---------------------------------------------------

# 导入你自己的核心算法模块
import config
from utils import file_utils, image_utils
from processing.stereo_matcher import StereoMatcher
from processing.reconstructor import Reconstructor

@celery_app.task(bind=True)
def run_stereo_reconstruction_task(self, left_path, right_path):
    """
    这是一个 Celery 后台任务。
    它包含了所有耗时的计算逻辑。
    `bind=True` 允许我们访问 `self` 来更新任务状态。
    """
    try:
        # --- 1. 更新初始状态 ---
        self.update_state(state='PROGRESS', meta={'status': '加载标定参数...', 'progress': 10})
        stereo_params = file_utils.load_stereo_params(config.CAMERA_PARAMS_PATH)
        if stereo_params is None:
            raise ValueError('相机标定文件未找到!')

        # --- 2. 读取图像 ---
        self.update_state(state='PROGRESS', meta={'status': '读取图像...', 'progress': 20})
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        if left_img is None or right_img is None:
            raise ValueError('无法读取上传的图像文件')

        # --- 3. 立体校正 ---
        self.update_state(state='PROGRESS', meta={'status': '执行立体校正...', 'progress': 40})
        left_rectified, right_rectified, Q = image_utils.rectify_stereo_pair(left_img, right_img, stereo_params)

        # --- 4. 计算视差 ---
        self.update_state(state='PROGRESS', meta={'status': '计算视差图...', 'progress': 60})
        matcher = StereoMatcher()
        disparity_map = matcher.compute_disparity(left_rectified, right_rectified)

        # --- 5. 生成结果图像 ---
        self.update_state(state='PROGRESS', meta={'status': '生成结果图像...', 'progress': 80})
        disp_to_show = cv2.normalize((disparity_map.astype(np.float32) / 16.0 - config.SGBM_MIN_DISPARITY) / config.SGBM_NUM_DISPARITIES, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        colormap_disp = cv2.applyColorMap(disp_to_show, cv2.COLORMAP_JET)

        result_filename = f"result_{os.path.basename(left_path)}"
        result_folder = os.path.join(PROJECT_ROOT, 'web_app', 'static', 'results')
        result_path = os.path.join(result_folder, result_filename)
        cv2.imwrite(result_path, colormap_disp)

        # --- 6. 任务成功 ---
        result_image_url = f'/static/results/{result_filename}'
        return {'status': '任务完成!', 'progress': 100, 'result_image': result_image_url}


    except Exception as e:
        # --- 任务失败 ---
        raise