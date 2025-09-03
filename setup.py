"""
light_kinfer 项目的安装配置文件

使用 pip install -e . 可以以开发模式安装，
这样可以直接导入 light_kinfer 包而无需修改 sys.path
"""

from setuptools import setup, find_packages

setup(
    name="light_kinfer",
    version="0.1.0",
    description="轻量级模型推理框架",
    author="Kaster",
    author_email="Kastermist@gmail.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "transformers",
        "numpy",
        # 添加其他依赖
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ]
    },
    entry_points={
        "console_scripts": [
            # 如果有命令行工具，可以在这里定义
        ],
    },
)
