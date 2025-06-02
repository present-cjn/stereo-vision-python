import config
import os
from calibration.calibrator import StereoCalibrator


def setup_environment():
    """
    负责程序运行前所有的环境准备工作，比如创建输出文件夹。
    """
    print("Setting up environment...")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    print("Output directory ensured.")


def main():
    setup_environment()

    print("Project setup is working!")
    print(f"Loading calibration images from: {config.CALIBRATION_IMAGE_DIR}")
    calibrator = StereoCalibrator()
    calibrator.run(config.CALIBRATION_IMAGE_DIR)


if __name__ == "__main__":
    main()