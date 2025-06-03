import os
import cv2

# 文件路径
CALIBRATION_IMAGE_DIR = "data/calibration_images/" # 标定用图片路径
TEST_IMAGE_DIR = "data/test_images/"
OUTPUT_DIR = "output/"
CAMERA_PARAMS_PATH = os.path.join(OUTPUT_DIR, "stereo_params.yml")


# 标定相关参数
Image_Number = 13
CHESSBOARD_SIZE = (11, 8)  # (内角点数量 a, 内角点数量 b)
SQUARE_SIZE_MM = 25        # 棋盘格尺寸（毫米）
IMAGE_SIZE = (640, 480)    # 图片分辨率

VISUALIZE_STEPS = False

MONO_CALIB_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
STEREO_CALIB_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
