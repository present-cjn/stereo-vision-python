import config
import os
from calibration.calibrator import StereoCalibrator
import argparse
from utils import file_utils, image_utils
import cv2
from visualization import visualizer
from processing.stereo_matcher import StereoMatcher
from processing.reconstructor import Reconstructor


def setup_environment():
    """负责程序运行前所有的环境准备工作，比如创建输出文件夹。"""
    print("Setting up environment...")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    print("Output directory ensured.")

# --- 创建不同的函数来处理不同的任务 ---

def handle_calibration(args):
    """处理标定任务的函数"""
    print("\n--- Running Calibration Task ---")
    if args.corners:
        print("Using chessboard size from command line.")
        try:
            corners_width, corners_height = map(int, args.corners.split(','))
            chessboard_size = (corners_width, corners_height)
        except ValueError:
            print(f"Error: Invalid format for --corners: '{args.corners}'. Please use 'width,height'.")
            return
    else:
        print("Using default chessboard size from config.py.")
        chessboard_size = config.CHESSBOARD_SIZE

    if args.size is not None:
        print("Using square size from command line.")
        square_size_mm = args.size
    else:
        print("Using default square size from config.py.")
        square_size_mm = config.SQUARE_SIZE_MM

    calibrator = StereoCalibrator(chessboard_size=chessboard_size, square_size=square_size_mm)
    calibrator.run(config.CALIBRATION_IMAGE_DIR)
    print("\nCalibration task finished.")

def handle_run_application(args):
    """处理核心应用（立体匹配等）任务的函数"""
    print("Loading calibration parameters...")
    stereo_params = file_utils.load_stereo_params(config.CAMERA_PARAMS_PATH)
    if stereo_params is None:
        print(f"Error: Calibration parameters not found at {config.CAMERA_PARAMS_PATH}. Please run the 'calibrate' command first.")
        return

    print("Loading test images...")
    left_img = cv2.imread(config.TEST_IMAGE_LEFT_PATH)
    right_img = cv2.imread(config.TEST_IMAGE_RIGHT_PATH)
    if left_img is None or right_img is None:
        print("Error: Could not load test images. Please check the paths in config.py.")
        return

    print("Performing stereo matching...")
    left_rectified, right_rectified, Q = image_utils.rectify_stereo_pair(left_img, right_img, stereo_params)

    # 增加一个可视化步骤，来检查校正效果
    if config.VERBOSE_MODE:
        # 这个函数需要你添加到 visualizer.py 中
        visualizer.show_rectified_pair(left_rectified, right_rectified)

    # 创建匹配器并计算视差图
    print("Computing disparity map...")
    matcher = StereoMatcher() # Matcher会从config加载SGBM参数
    disparity_map = matcher.compute_disparity(left_rectified, right_rectified)

    # 可视化最终的视差图
    if config.VERBOSE_MODE:
        print("Visualizing disparity map...")
        visualizer.show_disparity_map(
            disparity_map,
            config.SGBM_MIN_DISPARITY,
            config.SGBM_NUM_DISPARITIES
        )

    print("\nStereo matching application finished successfully.")

    # --- 三维重建 ---
    print("\n--- Performing 3D Reconstruction ---")
    reconstructor = Reconstructor()
    # 接收两种返回结果
    points_3D_matrix, (points_filtered, colors_filtered) = reconstructor.reconstruct(disparity_map, left_rectified, Q)

    # --- 保存点云 (使用过滤后的数据) ---
    print("\n--- Saving Point Cloud ---")
    # (可选) 在这里进行降采样
    downsample_factor = config.POINT_CLOUD_DOWNSAMPLE_FACTOR
    if downsample_factor > 1:
        points_to_save = points_filtered[::downsample_factor]
        colors_to_save = colors_filtered[::downsample_factor]
    else:
        points_to_save = points_filtered
        colors_to_save = colors_filtered


    # --- 可视化 ---
    print("\n--- Visualizing Final Output ---")
    visualizer.show_interactive_depth_map(
        disparity_map,
        left_rectified,
        points_3D_matrix,
        config.SGBM_MIN_DISPARITY,
        config.SGBM_NUM_DISPARITIES
    )

    if args.view_3d:
        print("\n--- Additionally visualizing 3D Point Cloud ---")
        file_utils.save_point_cloud(config.POINT_CLOUD_PATH, points_to_save, colors_to_save)
        visualizer.show_point_cloud(config.POINT_CLOUD_PATH)

    print("\nFull stereo vision pipeline finished successfully.")


def main():
    parser = argparse.ArgumentParser(description="A Stereo Vision Project.")

    # --- 定义全局参数---
    # 这个参数现在对所有子命令都有效。
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Enable verbose mode to show intermediate visualization steps for any task."
    )

    # 创建子命令解析器
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    # 创建 'calibrate' 命令
    parser_calibrate = subparsers.add_parser('calibrate', help='Run the stereo camera calibration process.')
    parser_calibrate.add_argument(
        '-c', '--corners',
        type=str,
        default=None,
        help=f"Chessboard inner corners 'width,height'. Overrides the default in config.py ({config.CHESSBOARD_SIZE})."
    )
    parser_calibrate.add_argument(
        '-s', '--size',
        type=float,
        default=None,
        help=f"Side length of a chessboard square in mm. Overrides the default in config.py ({config.SQUARE_SIZE_MM})."
    )
    parser_calibrate.set_defaults(func=handle_calibration)

    # 创建 'run' 命令
    parser_run = subparsers.add_parser('run', help='Run the main stereo matching application using existing calibration.')
    # 添加一个 --view-3d 参数，可以选择为输出点云图
    parser_run.add_argument(
        '--view-3d',
        action='store_true',
        help="Additionally, visualize the generated point cloud in 3D using Open3D."
    )
    parser_run.set_defaults(func=handle_run_application)

    # 解析命令行参数
    args = parser.parse_args()

    # --- 根据命令行参数，设置全局的配置状态 ---
    if args.verbose:
        config.VERBOSE_MODE = True
        print("Verbose mode is enabled.")

    # --- 根据解析出的命令，调用对应的处理函数 ---
    setup_environment()
    if args.command == 'run':
        args.func(args)
    else:
        args.func(args)

if __name__ == "__main__":
    main()