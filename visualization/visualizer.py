import cv2
import numpy as np


def display_chessboard_corners(image_left, ret_left, corners_left,
                               image_right, ret_right, corners_right,
                               chessboard_size):

    display_left = image_left.copy()
    display_right = image_right.copy()

    if ret_left:
        cv2.drawChessboardCorners(display_left, chessboard_size, corners_left, ret_left)
    if ret_right:
        cv2.drawChessboardCorners(display_right, chessboard_size, corners_right, ret_right)

    # 拼接图像用于显示
    h, w = display_left.shape[:2]
    display_image = np.zeros((h, w * 2, 3), dtype=np.uint8)
    display_image[:, :w] = display_left
    display_image[:, w:] = display_right

    # 调整窗口大小以适应屏幕
    window_name = 'Stereo Pair with Detected Corners'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, display_image)

    print("Press any key to continue to the next pair, or 'q' to quit.")
    key = _wait_for_key_or_window_close(window_name)

    return key

def show_disparity_map(disparity_map, min_disp, num_disp):
    """
    将原始视差图归一化并以伪彩色显示。

    Args:
        disparity_map (np.ndarray): CV_16S 格式的原始视差图。
        min_disp (int): SGBM的最小视差。
        num_disp (int): SGBM的视差范围。
    """
    # 将视差图转换为0-255范围的8位图像
    # 注意：disparity_map 的值是 16 * 真实视差，所以要除以 16
    disp_to_show = (disparity_map.astype(np.float32) / 16.0 - min_disp) / num_disp
    disp_to_show = cv2.normalize(disp_to_show, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 应用伪彩色映射，使其更易于观察
    colormap_disp = cv2.applyColorMap(disp_to_show, cv2.COLORMAP_JET)

    window_name = "Disparity Map"
    cv2.imshow(window_name, colormap_disp)

    _wait_for_key_or_window_close(window_name)

def show_rectified_pair(left_rectified, right_rectified):
    """
    将校正后的左右图像并排显示，并画上水平线以便观察。
    """
    # 获取图像尺寸
    h, w = left_rectified.shape[:2]

    # 创建一个能容纳两张图片的新画布
    combined_image = np.zeros((h, w * 2, 3), dtype=np.uint8)
    combined_image[:, :w] = left_rectified
    combined_image[:, w:] = right_rectified

    # 在图像上画上等间隔的水平线，方便检查对齐情况
    for i in range(1, 10):
        y = int(h / 10 * i)
        cv2.line(combined_image, (0, y), (w * 2, y), (0, 255, 0), 1)

    window_name = "Rectified Stereo Pair"
    cv2.imshow(window_name, combined_image)

    _wait_for_key_or_window_close(window_name)

# --- Internal Helper Functions ---
def _wait_for_key_or_window_close(window_name):
    """
    [内部辅助函数] 等待用户按键或关闭指定窗口。
    """
    print(f"Displaying window '{window_name}'. Press any key or close the window to continue.")

    key_pressed = -1 # 默认值，表示没有有效按键

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key != 255:
            key_pressed = key
            break

        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                key_pressed = -1
                break
        except cv2.error:
            # 当窗口被非常快地关闭时，getWindowProperty 可能会在窗口对象销毁后被调用，导致OpenCV错误。
            # 捕获这个错误并安全地退出循环。
            key_pressed = -1
            break

    cv2.destroyWindow(window_name)
    return key_pressed