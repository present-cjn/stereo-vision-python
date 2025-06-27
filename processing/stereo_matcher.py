# processing/stereo_matcher.py
import cv2
import numpy as np
import config


class StereoMatcher:
    def __init__(self):
        """
        初始化SGBM匹配器，并从config加载参数。
        """
        print("Initializing Stereo SGBM Matcher...")
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=config.SGBM_MIN_DISPARITY,
            numDisparities=config.SGBM_NUM_DISPARITIES,
            blockSize=config.SGBM_BLOCK_SIZE,
            P1=config.SGBM_P1,
            P2=config.SGBM_P2,
            disp12MaxDiff=config.SGBM_DISP12_MAX_DIFF,
            preFilterCap=config.SGBM_PRE_FILTER_CAP,
            uniquenessRatio=config.SGBM_UNIQUENESS_RATIO,
            speckleWindowSize=config.SGBM_SPECKLE_WINDOW_SIZE,
            speckleRange=config.SGBM_SPECKLE_RANGE,
            mode=config.SGBM_MODE
        )

    def compute_disparity(self, left_rectified_img, right_rectified_img):
        """
        计算视差图。

        Args:
            left_rectified_img (np.ndarray): 校正后的左图像 (CV_8U)。
            right_rectified_img (np.ndarray): 校正后的右图像 (CV_8U)。

        Returns:
            np.ndarray: 视差图 (CV_16S)。
        """
        print("Computing disparity map...")
        # SGBM算法要求输入灰度图
        gray_left = cv2.cvtColor(left_rectified_img, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rectified_img, cv2.COLOR_BGR2GRAY)

        disparity_map = self.matcher.compute(gray_left, gray_right)

        # 视差图的原始值范围比较大，且为有符号16位整数 (CV_16S)
        # 后面可视化时需要进行归一化
        print("Disparity map computation complete.")
        return disparity_map