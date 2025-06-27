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
            criteria=config.MONO_CALIB_CRITERIA
        )

        # 检查单目标定的质量
        assert ret < 5.0, f"{camera_name} camera reprojection error is too high: {ret}"
        print(f"  - {camera_name} camera calibrated with reprojection error: {ret}")

        return K, D, ret

    @staticmethod
    def _calibrate_stereo_relationship(obj_points, img_points_l, img_points_r, K1, D1, K2, D2, img_size):
        """
        在已知各自内参的情况下，计算双目相机之间的旋转和平移。
        :param obj_points: 世界坐标系下的点
        :param img_points_l: 左摄像头图像上的角点
        :param img_points_r: 右摄像头图像上的角点
        :param K1:
        :param D1:
        :param K2:
        :param D2:
        :param img_size: 图像尺寸（width, height）
        :return:
        """
        print("\nPerforming stereo calibration to find the relationship between cameras...")

        flags = config.STEREO_CALIB_FLAGS

        ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            obj_points, img_points_l, img_points_r,
            K1, D1,  # 将我们单目标定得到的精确内参传入
            K2, D2,  # 将我们单目标定得到的精确内参传入
            img_size,
            flags=flags,
            criteria=config.STEREO_CALIB_CRITERIA  # 可以和单目标定的 criteria 不同
        )

        assert ret < 5.0, f"Stereo calibration reprojection error is too high: {ret}"
        print(f"  - Stereo relationship calibrated with reprojection error: {ret}")

        # 将所有最终参数打包
        stereo_params = {
            "K1": K1, "D1": D1, "K2": K2, "D2": D2,
            "R": R, "T": T, "E": E, "F": F, "reprojection_error": ret
        }
        return stereo_params

    def _find_corners_in_all_images(self, image_pairs: list):
        """ 存储检测到的角点"""
        object_points = []  # 存储世界坐标
        image_points_left = []  # 存储左相机图像点
        image_points_right = []  # 存储右相机图像点
        image_size = None  # 将在第一张有效图片中获取

        """遍历图像找角点"""
        for left_image_path, right_image_path in image_pairs:
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
            # --- 为左图添加 NORMALIZE_IMAGE 标志 ---
            find_flags_l = cv2.CALIB_CB_NORMALIZE_IMAGE
            ret_left, corners_left = cv2.findChessboardCorners(gray_left, self.chessboard_size, flags=find_flags_l)
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
                corners_left_subpix = cv2.cornerSubPix(gray_left, corners_left, (3, 3), (-1, -1), criteria=config.SUBPIX_CRITERIA)
                corners_right_subpix = cv2.cornerSubPix(gray_right, corners_right, (3, 3), (-1, -1), criteria=config.SUBPIX_CRITERIA)

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
            criteria=config.STEREO_CALIB_CRITERIA
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

    def run(self, image_source):
        """
        执行完整的标定流程并保存结果。
        image_source (str or list): 如果是字符串，则视为目录路径；
                                    如果是列表，则视为图片对的列表。
        """
        try:
            if isinstance(image_source, str): # 传入的是目录
                """加载图片"""
                images_left = sorted(glob.glob(os.path.join(image_source, 'leftPic*.jpg')), key=file_utils.natural_sort_key)
                images_right = sorted(glob.glob(os.path.join(image_source, 'rightPic*.jpg')), key=file_utils.natural_sort_key)
                assert len(images_left) == len(images_right), "Number of left and right images must be equal"
                image_pairs = list(zip(images_left, images_right))
            elif isinstance(image_source, list): # 传入的是列表
                image_pairs = image_source
            else:
                raise TypeError("image_source must be a directory path (str) or a list of pairs.")

            # --- 第零步：寻找所有角 ---
            print("Step 0: Finding chessboard corners in all images...")
            obj_points, img_points_l, img_points_r, img_size = self._find_corners_in_all_images(image_pairs)

            # --- 第一步：分别标定左右相机 ---
            print("\nStep 1: Calibrating each camera individually...")
            K1, D1, reproj_error_L = self._calibrate_single_camera(obj_points, img_points_l, img_size, "Left")
            K2, D2, reproj_error_R = self._calibrate_single_camera(obj_points, img_points_r, img_size, "Right")

            # --- 第二步：标定双目关系 ---
            print("\nStep 2: Calibrating the stereo rig relationship...")
            stereo_params = self._calibrate_stereo_relationship(
                obj_points, img_points_l, img_points_r, K1, D1, K2, D2, img_size
            )

            # --- 最终步骤：整合、保存、返回 ---
            # 把单目标定的误差也加进去，便于诊断
            stereo_params['reprojection_error_L'] = reproj_error_L
            stereo_params['reprojection_error_R'] = reproj_error_R

            print("\nCalibration process completed successfully!")
            file_utils.save_stereo_params(config.CAMERA_PARAMS_PATH, stereo_params)

            return stereo_params

        except (FileNotFoundError, ValueError, InterruptedError, AssertionError) as e:
            print(f"\nAn error occurred during calibration: {e}")
            print("Calibration process failed.")
            return None
