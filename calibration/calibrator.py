import glob
import config
import os
import cv2
import numpy as np
from visualization import visualizer
from utils import file_utils


class StereoCalibrator:
    def __init__(self):
        self.chessboard_size = config.CHESSBOARD_SIZE
        self.square_size = config.SQUARE_SIZE_MM

        """准备 objectPoints"""
        self.objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        self.objp = self.objp * self.square_size


    @staticmethod
    def _calibrate_single_camera(obj_points, img_points, img_size, camera_name: str):
        """
        单目标定函数
        :param obj_points: 世界坐标系下的点
        :param img_points: 图像上的角点
        :param img_size: 图像尺寸（width, height）
        :param camera_name: 用于打印日志的相机名字 (e.g., "Left" or "Right")。
        :return:一个包含 K, D, 和重投影误差的元组。
        """
        print(f"\nPerforming monocular calibration for {camera_name} camera...")

        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
            obj_points,
            img_points,
            img_size,
            None,
            None,
            criteria=config.MONO_CALIB_CRITERIA # 建议在config中为单目标定设置独立的criteria
        )

        # 检查单目标定的质量
        assert ret < 1.0, f"{camera_name} camera reprojection error is too high: {ret}"
        print(f"  - {camera_name} camera calibrated with reprojection error: {ret}")

        return K, D, ret

    def _find_corners_in_all_images(self, image_dir: str):
        """ 存储检测到的角点"""
        object_points = []  # 存储世界坐标
        image_points_left = []  # 存储左相机图像点
        image_points_right = []  # 存储右相机图像点
        image_size = None  # 将在第一张有效图片中获取

        """加载图片"""
        left_image_path_pattern = os.path.join(image_dir, 'leftPic*.jpg')
        right_image_path_pattern = os.path.join(image_dir, 'rightPic*.jpg')

        images_left = sorted(glob.glob(left_image_path_pattern), key=file_utils.natural_sort_key)
        images_right = sorted(glob.glob(right_image_path_pattern), key=file_utils.natural_sort_key)

        assert len(images_left) == len(images_right), "Number of left and right images must be equal"

        """遍历图像找角点"""
        for left_image_path, right_image_path in zip(images_left, images_right):
            image_left = cv2.imread(left_image_path)
            image_right = cv2.imread(right_image_path)
            # 转换为灰度图
            gray_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
            """获取image_size并判断所有图片尺寸"""
            if image_size is None:
                image_size = gray_left.shape[::-1]

            assert gray_left.shape[::-1] == image_size, \
                f"Image size mismatch! Expected {image_size}, but got {gray_left.shape[::-1]} in {image_left}"

            assert gray_right.shape[::-1] == image_size, \
                f"Image size mismatch! Expected {image_size}, but got {gray_right.shape[::-1]} in {image_right}"

            # 查找棋盘格角点
            ret_left, corners_left = cv2.findChessboardCorners(gray_left, self.chessboard_size, None)
            ret_right, corners_right = cv2.findChessboardCorners(gray_right, self.chessboard_size, None)

            if config.VISUALIZE_STEPS:
                key = visualizer.display_chessboard_corners(
                    image_left, ret_left, corners_left,
                    image_right, ret_right, corners_right,
                    self.chessboard_size
                )
                # 如果用户按了 'q'，则退出标定
                if key == ord('q'):
                    print("Calibration cancelled by user.")
                    return None  # 或者抛出异常

            # 如果左右图像都成功找到了角点
            if ret_left and ret_right:
                # 亚像素精度优化
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_left_subpix = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
                corners_right_subpix = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

                object_points.append(self.objp)
                image_points_left.append(corners_left_subpix)
                image_points_right.append(corners_right_subpix)

        if not object_points:
            raise ValueError("Could not find chessboard corners in any of the image pairs.")
        if image_size is None:
            raise ValueError("Could not determine image size. No valid images found.")
        return object_points, image_points_left, image_points_right, image_size

    @staticmethod
    def _perform_calibration(object_points, image_points_left, image_points_right, image_size):
        """双目标定"""
        cameraMatrix1 = np.eye(3, dtype=np.float64)
        distCoeffs1 = np.zeros(5, dtype=np.float64)
        cameraMatrix2 = np.eye(3, dtype=np.float64)
        distCoeffs2 = np.zeros(5, dtype=np.float64)

        print("Starting stereo calibration... This may take a while.")
        ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            object_points, image_points_left, image_points_right,
            cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
            image_size, R=None, T=None,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS,  # Or cv2.CALIB_FIX_INTRINSIC if you have pre-calibrated values
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        )
        assert K1.shape == (3, 3), "K1 matrix shape is incorrect!"
        assert R.shape == (3, 3), "R rotation matrix shape is incorrect!"
        assert T.shape == (3, 1), "T translation vector shape is incorrect!"
        assert ret < 6.0, f"Reprojection error {ret} is too high! Calibration likely failed."

        # 将所有结果打包成一个字典
        stereo_params = {
            "K1": K1, "D1": D1, "K2": K2, "D2": D2,
            "R": R, "T": T, "E": E, "F": F
        }
        return stereo_params

    def run(self, image_dir: str):
        """
        执行完整的标定流程并保存结果。
        """
        try:
            # 1.找到图片并计算角点
            obj_points, img_points_l, img_points_r, img_size = self._find_corners_in_all_images(image_dir)
            # 2.执行标定计算
            stereo_params = self._perform_calibration(obj_points, img_points_l, img_points_r, img_size)
            # 3.保存结果
            file_utils.save_stereo_params(config.CAMERA_PARAMS_PATH, stereo_params)

            return stereo_params
        except (FileNotFoundError, ValueError, InterruptedError) as e:
            print(f"An error occurred during calibration: {e}")
            return None
