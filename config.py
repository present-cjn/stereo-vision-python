import os
import cv2

# --- Path configurations ---
# project root directory, determined by the location of this config file.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CALIBRATION_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data/calibration_images/")
TEST_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data/test_images/")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output/")
CAMERA_PARAMS_PATH = os.path.join(OUTPUT_DIR, "stereo_params.yml")


# ---Calibration Target Parameters---
# Image_Number = 13
CHESSBOARD_SIZE = (11, 8)  # (内角点数量 a, 内角点数量 b)
SQUARE_SIZE_MM = 25        # 棋盘格尺寸（毫米）
# IMAGE_SIZE = (640, 480)    # 图片分辨率

# --- Runtime Control Flags ---
# 显示棋盘角点的过程图，设置为True表示显示
VISUALIZE_STEPS = False

# --- Algorithm Hyperparameters ---
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
MONO_CALIB_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
STEREO_CALIB_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
