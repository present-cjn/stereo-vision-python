import cv2
import re

# TODO(cjn): Refactor this to use a Pydantic model for data validation.
# This will prevent silent errors from malformed or type-incorrect data in the YAML file.
# This should be addressed after the two-step calibration logic is complete.
def save_stereo_params(path, stereo_params):
    """
    使用 OpenCV 的 FileStorage 保存双目标定参数。
    :param path: yml文件的保存路径。
    :param stereo_params: 包含所有相机参数的字典。
    :return:
    """
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    for key, value in stereo_params.items():
        fs.write(key, value)
    fs.release()
    print(f"Stereo parameters saved to {path}")


def load_stereo_params(path):
    """
    从 yml 文件加载双目标定参数。
    :param path:yml文件的路径。
    :return: params
    """
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Could not open stereo parameters file: {path}")

    params = {}
    # 动态读取所有节点
    root = fs.root()
    for key in root.keys():
        node = fs.getNode(key)
        if node.isReal():
            params[key] = node.real()
            if node.isInt():
                params[key] = int(params[key])

        elif node.isString():
            params[key] = node.string()

        elif node.isNone():
            params[key] = None

        else:
            params[key] = fs.getNode(key).mat()
    fs.release()
    print(f"Stereo parameters loaded from {path}")
    return params

def save_point_cloud(path, points_3D, colors):
    """使用 Open3D 将点云保存为 .ply 文件。"""
    try:
        import open3d as o3d
    except ImportError:
        print("\n[Warning] Open3D is not installed, cannot save point cloud.")
        print("To install, run: pip install open3d")
        return
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3D)
    # Open3D 需要的颜色值是 0-1 范围的浮点数
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    o3d.io.write_point_cloud(path, pcd)
    print(f"Point cloud saved to {path}")

def natural_sort_key(s):
    """
    一个用于 sorted() 函数的 key 函数，实现自然排序。
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]