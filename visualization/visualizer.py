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
    key = cv2.waitKey(0)  # 无限期等待按键

    # 在调用后关闭窗口，避免窗口堆叠
    cv2.destroyWindow(window_name)

    return key
