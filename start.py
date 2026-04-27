"""一键启动疲劳驾驶检测系统 - 双击此文件即可运行"""

import os
import sys
import time
import threading
import webbrowser

# 确保工作目录为脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 添加项目根目录到 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def open_browser():
    """延迟 1.5 秒后自动打开浏览器"""
    time.sleep(1.5)
    webbrowser.open("http://localhost:5000")


def check_dependencies():
    """检查必要依赖是否已安装"""
    missing = []
    for pkg, import_name in [
        ("flask", "flask"),
        ("opencv-python", "cv2"),
        ("mediapipe", "mediapipe"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if missing:
        print("=" * 50)
        print("缺少以下依赖，正在自动安装...")
        print(", ".join(missing))
        print("=" * 50)
        import subprocess
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", *missing
        ])
        print("依赖安装完成！\n")


if __name__ == "__main__":
    print("=" * 50)
    print("  疲劳驾驶检测系统 - 启动中...")
    print("=" * 50)

    # 检查依赖
    check_dependencies()

    # 自动打开浏览器
    threading.Thread(target=open_browser, daemon=True).start()

    print("\n系统已启动！浏览器将自动打开...")
    print("访问地址: http://localhost:5000")
    print("按 Ctrl+C 停止服务\n")

    # 启动 Flask
    from web_app import app
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
