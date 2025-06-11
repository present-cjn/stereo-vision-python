# tests/test_calibration.py
import os
import pytest
from calibration.calibrator import StereoCalibrator
import config

def test_calibration_produces_valid_file():
    # 准备：确保没有旧的参数文件
    if os.path.exists(config.CAMERA_PARAMS_PATH):
        os.remove(config.CAMERA_PARAMS_PATH)

    # 执行：运行标定器
    calibrator = StereoCalibrator()
    # 这里我们假设标定会成功，如果失败会抛出异常，pytest会自动捕获
    calibrator.run(config.CALIBRATION_IMAGE_DIR)

    # 验证：
    # 1. 检查文件是否已创建
    assert os.path.exists(config.CAMERA_PARAMS_PATH), "Calibration did not create the parameter file."

    # 2. （更进一步）可以加载文件并检查内容
    from utils import file_utils
    params = file_utils.load_stereo_params(config.CAMERA_PARAMS_PATH)
    assert "reprojection_error_L" in params
    assert "reprojection_error_R" in params
