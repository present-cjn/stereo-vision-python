import sys
import os
from flask import Flask

# --- 关键的路径设置 ---
# 将项目根目录添加到 Python 的模块搜索路径中
# 这样我们才能在 web_app 内部，导入像 processing, calibration 这样的模块
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
# --------------------

# from processing.stereo_matcher import StereoMatcher # 我们稍后会用到

# 初始化 Flask 应用
app = Flask(__name__)

@app.route('/')
def hello_world():
    """定义了网站根目录的行为"""
    return "<h1>你好，Web 应用！</h1><p>你的 Flask 服务器已经成功运行。</p>"

if __name__ == '__main__':
    # 启动用于开发的 Web 服务器
    # debug=True 表示每次修改代码后，服务器会自动重启
    app.run(debug=True, port=5000)