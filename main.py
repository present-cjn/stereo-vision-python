import config
import os
from calibration.calibrator import StereoCalibrator
import argparse


def setup_environment():
    """负责程序运行前所有的环境准备工作，比如创建输出文件夹。"""
    print("Setting up environment...")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    print("Output directory ensured.")

# --- 创建不同的函数来处理不同的任务 ---

def handle_calibration():
    """处理标定任务的函数"""
    print("\n--- Running Calibration Task ---")
    calibrator = StereoCalibrator()
    calibrator.run(config.CALIBRATION_IMAGE_DIR)
    print("\nCalibration task finished.")

def handle_run_application():
    """处理核心应用（立体匹配等）任务的函数"""
    print("\n--- Running Main Application ---")
    # TODO: 下一步要填充的立体匹配流程
    print("Loading stereo parameters...")
    print("Loading test images...")
    print("Performing stereo matching...")
    print("\nMain application task finished.")


def main():
    parser = argparse.ArgumentParser(description="A Stereo Vision Project.")

    # 创建子命令解析器
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    # 创建 'calibrate' 命令
    parser_calibrate = subparsers.add_parser('calibrate', help='Run the stereo camera calibration process.')
    parser_calibrate.set_defaults(func=handle_calibration)

    # 创建 'run' 命令
    parser_run = subparsers.add_parser('run', help='Run the main stereo matching application using existing calibration.')
    parser_run.set_defaults(func=handle_run_application)

    # 解析命令行参数
    args = parser.parse_args()

    # --- 根据解析出的命令，调用对应的处理函数 ---
    setup_environment()
    args.func()

if __name__ == "__main__":
    main()