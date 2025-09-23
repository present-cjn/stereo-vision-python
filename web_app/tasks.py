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
def run_stereo_reconstruction_task(self, left_path, right_path, params_path):
    """
    接收左右图像和标定文件路径，执行后台重建任务。
    """
    try:
        # --- 1. 更新初始状态 ---
        self.update_state(state='PROGRESS', meta={'status': '加载标定参数...', 'progress': 10})

        # --- 核心修改：从用户上传的指定路径加载参数 ---
        stereo_params = file_utils.load_stereo_params(params_path)
        if stereo_params is None:
            raise ValueError(f'无法从 {os.path.basename(params_path)} 加载或解析标定参数!')

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
        disp_to_show = cv2.normalize(
            (disparity_map.astype(np.float32) / 16.0 - config.SGBM_MIN_DISPARITY) / config.SGBM_NUM_DISPARITIES, None,
            alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        colormap_disp = cv2.applyColorMap(disp_to_show, cv2.COLORMAP_JET)

        # 为结果文件创建一个唯一的名称，避免浏览器缓存问题
        unique_id = os.path.basename(os.path.dirname(left_path))
        result_filename = f"result_{unique_id}.png"
        result_folder = os.path.join(PROJECT_ROOT, 'web_app', 'static', 'results')
        result_path = os.path.join(result_folder, result_filename)
        cv2.imwrite(result_path, colormap_disp)

        # --- 6. 任务成功 ---
        result_image_url = f'/static/results/{result_filename}'
        return {'status': '任务完成!', 'progress': 100, 'result_image': result_image_url}

    except Exception as e:
        # 重新抛出异常，让 Celery 自动处理失败状态
        raise

# 导入 StereoCalibrator
from calibration.calibrator import StereoCalibrator


@celery_app.task(bind=True)
def run_calibration_task(self, image_dir, chessboard_size_str, square_size_mm):
    """
    后台执行相机标定任务。
    """
    try:
        # --- 1. 解析参数 ---
        self.update_state(state='PROGRESS', meta={'status': '解析参数...', 'progress': 5})
        try:
            width, height = map(int, chessboard_size_str.split(','))
            chessboard_size = (width, height)
        except ValueError:
            raise ValueError(f"内角点数格式错误: '{chessboard_size_str}'")

        # --- 2. 实例化并运行标定器 ---
        # 我们可以通过更新状态来报告进度，但标定器本身没有分步进度
        self.update_state(state='PROGRESS', meta={'status': '正在进行标定计算，请耐心等待...', 'progress': 20})

        calibrator = StereoCalibrator(
            chessboard_size=chessboard_size,
            square_size=float(square_size_mm)
        )
        # 标定器会使用 config.CALIBRATION_IMAGE_DIR，我们需要确保上传的文件在那里
        # 注意：为了简化，我们让Web上传的图片也保存到这个公共目录
        stereo_params = calibrator.run(image_dir)

        if stereo_params is None:
            raise RuntimeError("标定失败，请检查图像质量或参数。")

        # --- 3. 任务成功 ---
        self.update_state(state='PROGRESS', meta={'status': '标定成功，正在生成结果...', 'progress': 90})

        reprojection_error = stereo_params.get('reprojection_error', 'N/A')

        # 返回结果：成功信息、误差和一个下载链接
        return {
            'status': '标定成功!',
            'progress': 100,
            'reprojection_error': f"{reprojection_error:.4f}",
            'download_url': '/download_params'  # 我们需要创建一个新的端点来处理下载
        }

    except Exception as e:
        self.update_state(state='FAILURE', meta={'status': str(e), 'progress': 0})
        raise