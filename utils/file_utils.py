import cv2
import re


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
        params[key] = fs.getNode(key).mat()
    fs.release()
    print(f"Stereo parameters loaded from {path}")
    return params


def natural_sort_key(s):
    """
    一个用于 sorted() 函数的 key 函数，实现自然排序。
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]