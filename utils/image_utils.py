# utils/image_utils.py
import cv2
import numpy as np


def rectify_stereo_pair(left_img, right_img, stereo_params):
    """
    使用标定参数对左右图像进行立体校正。

    Args:
        left_img (np.ndarray): 原始左图像。
        right_img (np.ndarray): 原始右图像。
        stereo_params (dict): 从文件加载的标定参数字典。

    Returns:
        A tuple containing:
        - left_rectified (np.ndarray): 校正后的左图像。
        - right_rectified (np.ndarray): 校正后的右图像。
        - Q (np.ndarray): 4x4 的视差转深度重投影矩阵。
    """
    print("Rectifying stereo image pair...")

    # 从参数字典中提取需要的矩阵
    K1 = stereo_params['K1']
    D1 = stereo_params['D1']
    K2 = stereo_params['K2']
    D2 = stereo_params['D2']
    R = stereo_params['R']
    T = stereo_params['T']

    # 获取图像尺寸
    height, width = left_img.shape[:2]
    image_size = (width, height)

    # --- 核心步骤: 执行 cv2.stereoRectify ---
    # 这个函数计算校正变换所需的旋转矩阵(R1, R2)、投影矩阵(P1, P2)和Q矩阵
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, T, alpha=0
    )

    # --- 计算校正所需的映射表 ---
    # alpha=0: 校正后图像无黑边，但会裁剪掉一部分像素
    # alpha=1: 保留所有原始像素，但校正后图像会有黑边
    left_map1, left_map2 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_16SC2)

    # --- 应用映射表，进行重映射 ---
    left_rectified = cv2.remap(left_img, left_map1, left_map2, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_img, right_map1, right_map2, cv2.INTER_LINEAR)

    print("Rectification complete.")
    return left_rectified, right_rectified, Q