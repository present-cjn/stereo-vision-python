import os
import cv2

# --- Path configurations ---
# project root directory, determined by the location of this config file.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CALIBRATION_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data/calibration_images/")

# 定义测试图片路径
TEST_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data/test_images/")
TEST_IMAGE_LEFT_PATH = os.path.join(TEST_IMAGE_DIR, "leftPic.jpg")
TEST_IMAGE_RIGHT_PATH = os.path.join(TEST_IMAGE_DIR,"rightPic.jpg")

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

# SGBM (Semi-Global Block Matching) Parameters
SGBM_MIN_DISPARITY = 0
SGBM_NUM_DISPARITIES = 128      # 视差搜索范围，必须是16的倍数
SGBM_BLOCK_SIZE = 5             # 匹配块大小，通常是奇数
SGBM_P1 = 8 * 3 * SGBM_BLOCK_SIZE**2  # 控制视差平滑度的参数 P1
SGBM_P2 = 32 * 3 * SGBM_BLOCK_SIZE**2 # 控制视差平滑度的参数 P2
SGBM_DISP12_MAX_DIFF = 1      # 左右视差图检查的最大差异
SGBM_PRE_FILTER_CAP = 63        # 预处理滤波器的截断值
SGBM_UNIQUENESS_RATIO = 10      # 唯一性检查的裕量
SGBM_SPECKLE_WINDOW_SIZE = 100  # 视差图后处理的散斑窗口大小
SGBM_SPECKLE_RANGE = 32         # 散斑窗口内的最大视差变化
SGBM_MODE = cv2.STEREO_SGBM_MODE_SGBM_3WAY # SGBM模式
