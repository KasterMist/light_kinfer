# light_kinfer 项目

轻量级模型推理框架

## 安装方法

### 开发模式安装（推荐）
```bash
cd /path/to/light_kinfer
pip install -e .
```

这样安装后，可以在任何地方导入 light_kinfer 包：
```python
from light_kinfer.models.model_config import LlamaConfig
from light_kinfer.utils.common import getProjectPath
```

### 项目结构
```
light_kinfer/
├── setup.py              # 包安装配置
├── light_kinfer/          # 主包目录
│   ├── __init__.py       # 包初始化文件
│   ├── models/           # 模型相关模块
│   │   ├── __init__.py
│   │   └── model_config.py
│   └── utils/            # 工具模块
│       ├── __init__.py
│       ├── common.py
│       └── config_convert.py
└── tests/                # 测试目录
    └── ...
```

## 使用方法

```python
# 方法1：直接运行模块
python -m light_kinfer.utils.config_convert

# 方法2：导入使用
from light_kinfer.utils.config_convert import convert_transformers_to_custom_config
```
