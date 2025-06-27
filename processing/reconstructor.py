# processing/reconstructor.py
import cv2
import numpy as np
import config

class Reconstructor:
    def __init__(self):
        print("Initializing Reconstructor...")

    def reconstruct(self, disparity_map, left_rectified_img, Q_matrix):
        """
        根据视差图、左校正图和Q矩阵重建三维点云。

        Args:
            disparity_map (np.ndarray): SGBM算法输出的原始视差图 (CV_16S)。
            left_rectified_img (np.ndarray): 用于给点云上色的左校正图 (CV_8U)。
            Q_matrix (np.ndarray): 4x4的重投影矩阵。

        Returns:
            A tuple containing:
            - points_3D (np.ndarray): N_points x 3 的点云坐标数组。
            - colors (np.ndarray): N_points x 3 的点云颜色数组 (RGB)。
        """
        print("Reconstructing 3D point cloud...")

        # --- 核心步骤: 使用 reprojectImageTo3D 将视差图转换为三维坐标 ---
        # 这个函数会返回一个与视差图同样大小的 HxWx3 图像
        # 其中每个像素(x,y)包含了三维坐标 (X, Y, Z)
        points_3D_matrix = cv2.reprojectImageTo3D(disparity_map, Q_matrix)

        # 提取颜色信息
        colors_matrix = cv2.cvtColor(left_rectified_img, cv2.COLOR_BGR2RGB)

        # --- 过滤无效点 ---
        # 1. reprojectImageTo3D 会对无效视差（值小于或等于0）的点生成一个很大的Z值。
        #    我们可以根据视差图本身来创建一个掩码(mask)，过滤掉这些无效点。
        #    OpenCV的视差值是原始视差的16倍，所以我们要找 > 0 的点。
        mask = disparity_map > disparity_map.min()

        # 2. 从矩阵中提取有效的点和颜色
        points_3D = points_3D_matrix[mask]
        colors = colors_matrix[mask]

        # 3. (可选) 过滤掉Z值过大（离相机太远）的点，这可以去除背景和噪点
        z_max_threshold = 1000.0 # 阈值：1000毫米 (1米)
        far_points_mask = points_3D[:, 2] < z_max_threshold
        points_3D = points_3D[far_points_mask]
        colors = colors[far_points_mask]

        # 4. (可选) 为了性能考虑，进行降采样
        config.POINT_CLOUD_DOWNSAMPLE_FACTOR = 4
        factor = config.POINT_CLOUD_DOWNSAMPLE_FACTOR
        if factor > 1:
            points_3D = points_3D[::factor]
            colors = colors[::factor]

        print(f"Reconstruction complete. Generated {len(points_3D)} points.")
        return points_3D, colors