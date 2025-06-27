import cv2
import numpy as np
import open3d as o3d


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

def show_point_cloud(ply_file_path):
    """加载并显示 .ply 格式的点云文件。"""
    print(f"Visualizing point cloud from {ply_file_path}...")
    pcd = o3d.io.read_point_cloud(ply_file_path)
    if not pcd.has_points():
        print("Could not read point cloud or point cloud is empty.")
        return

    # 创建一个可视化窗口并显示点云
    o3d.visualization.draw_geometries([pcd])


def show_interactive_depth_map(
        disparity_map,
        left_image_for_display,
        points_3D,
        min_disp,
        num_disp
):
    """
    创建一个交互式窗口，显示伪彩色的视差图，并在鼠标悬停处显示Z轴深度。

    Args:
        disparity_map (np.ndarray): 原始视差图 (CV_16S).
        left_image_for_display (np.ndarray): 用于在旁边显示的左相机图像。
        points_3D (np.ndarray): HxWx3 的三维点坐标矩阵。
        min_disp (int): SGBM的最小视差。
        num_disp (int): SGBM的视差范围。
    """
    print("\n--- Interactive Depth Map ---")
    print("Move mouse over the depth map to see distance. Press 'q' or close window to exit.")

    # 创建一个字典来在回调函数和主循环之间共享数据
    mouse_params = {'x': -1, 'y': -1}
    window_name = "Interactive Depth Map"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # --- 定义鼠标回调函数 ---
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            param['x'] = x
            param['y'] = y

    cv2.setMouseCallback(window_name, on_mouse, mouse_params)

    # --- 准备用于显示的伪彩色视差图 ---
    disp_to_show = (disparity_map.astype(np.float32) / 16.0 - min_disp) / num_disp
    disp_to_show = cv2.normalize(disp_to_show, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    colormap_disp = cv2.applyColorMap(disp_to_show, cv2.COLORMAP_JET)

    # 将左图和视差图拼接在一起显示
    h, w = left_image_for_display.shape[:2]
    combined_display = np.zeros((h, w * 2, 3), dtype=np.uint8)
    combined_display[:, :w] = left_image_for_display
    combined_display[:, w:] = colormap_disp

    while True:
        # 复制一份显示图像，以免永久地在上面画字
        display_copy = combined_display.copy()

        x, y = mouse_params['x'], mouse_params['y']

        # 检查鼠标是否在右侧的视差图区域内
        if w < x < w * 2 and 0 < y < h:
            # 获取对应的3D点
            point_3d = points_3D[y, x - w]
            px, py, pz = point_3d[0], point_3d[1], point_3d[2]

            # 准备要显示的文本
            # 我们只显示Z值（深度），单位是毫米(mm)，可以转换为米(m)
            # 过滤掉无效的深度值（通常Z值会非常大）
            if pz < 10000:  # 过滤掉10米以外的点
                distance_text = f"Distance: {pz / 1000:.2f} m"
            else:
                distance_text = "Distance: inf"

            # 在图像上绘制文本
            # 在鼠标位置附近显示
            cv2.putText(display_copy, distance_text, (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                        2)
            # 也可以在固定位置显示
            cv2.rectangle(display_copy, (0, 0), (w * 2, 30), (0, 0, 0), -1)  # 创建一个黑色背景条
            cv2.putText(display_copy, f"({x - w}, {y}) -> {distance_text}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

        cv2.imshow(window_name, display_copy)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 按 'q' 或 'ESC' 退出
            break

        # 检查窗口是否被关闭
        try:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break

    cv2.destroyAllWindows()

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