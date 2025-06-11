# tests/test_calibration_stability.py
import os
import glob
import pytest
import random
import numpy as np
from calibration.calibrator import StereoCalibrator
import config
from utils import file_utils

def test_calibration_is_stable_against_image_order():
    """
    验证标定算法对输入图片顺序不敏感。
    """
    # 执行：运行标定器
    calibrator = StereoCalibrator()

    # --- 1. 获取图片列表 ---
    image_dir = config.CALIBRATION_IMAGE_DIR
    left_paths = sorted(glob.glob(os.path.join(image_dir, 'leftPic*.jpg')), key=file_utils.natural_sort_key)
    right_paths = sorted(glob.glob(os.path.join(image_dir, 'rightPic*.jpg')), key=file_utils.natural_sort_key)
    image_pairs = list(zip(left_paths, right_paths))

    # --- 2. 运行一次基准标定，通过公共接口 run() ---
    print("\nRunning baseline calibration with sorted images...")
    baseline_params = calibrator.run(image_pairs)  # <--- 调用公共接口

    assert baseline_params is not None, "Baseline calibration failed"
    baseline_error = baseline_params['reprojection_error']
    print(f"Baseline reprojection error: {baseline_error}")

    # --- 3. 多次打乱顺序进行标定，并对比结果 ---
    num_runs = 5
    for i in range(num_runs):
        print(f"\nRunning randomized calibration run #{i + 1}/{num_runs}...")

        shuffled_pairs = image_pairs.copy()
        random.shuffle(shuffled_pairs)

        # --- 依然调用公共接口 run() ---
        shuffled_params = calibrator.run(shuffled_pairs)  # <--- 调用公共接口

        assert shuffled_params is not None, f"Shuffled run #{i + 1} failed"
        shuffled_error = shuffled_params['reprojection_error']
        print(f"Shuffled run #{i + 1} reprojection error: {shuffled_error}")

        # --- 4. 验证结果（这部分不变）---
        assert shuffled_error == pytest.approx(baseline_error, rel=1e-3)
        assert np.allclose(shuffled_params['K1'], baseline_params['K1'], rtol=1e-4)
