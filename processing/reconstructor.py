# processing/reconstructor.py
import cv2
import numpy as np
import config

class Reconstructor:
    def __init__(self):
        print("Initializing Reconstructor...")

    def reconstruct(self, disparity_map, left_rectified_img, Q_matrix):
        """
        [升级版] 返回两种形式的点云数据：
        1. 原始的、与图像对应的3D矩阵（用于交互式查找）。
        2. 经过过滤和清理的点列表（用于保存和3D可视化）。
        """
        # --- 生成原始的3D点矩阵 ---
        points_3D_matrix = cv2.reprojectImageTo3D(disparity_map, Q_matrix)
        colors_matrix = cv2.cvtColor(left_rectified_img, cv2.COLOR_BGR2RGB)

        # --- 过滤无效点，生成干净的点列表 ---
        mask = disparity_map > disparity_map.min()
        points_3D_filtered = points_3D_matrix[mask]
        colors_filtered = colors_matrix[mask]

        # (可选) 进一步过滤远点
        z_max_threshold = 1000.0
        far_points_mask = points_3D_filtered[:, 2] < z_max_threshold
        points_3D_filtered = points_3D_filtered[far_points_mask]
        colors_filtered = colors_filtered[far_points_mask]

        # --- 返回两种数据 ---
        # 注意：不在这里做降采样，降采样可以移到保存或显示之前，让数据更纯粹
        return points_3D_matrix, (points_3D_filtered, colors_filtered)
