import os


def get_model_name_from_path(model_path):
    """
    从模型路径中提取模型名称。

    Args:
        model_path (str): 模型路径，可能包含多个目录层级。

    Returns:
        str: 提取的模型名称。
        - 如果路径的最后一部分以 "checkpoint-" 开头，则返回 "上一级目录名_最后一部分"。
        - 否则，返回路径的最后一部分。

    示例:
        >>> get_model_name_from_path("/path/to/model/checkpoint-12345")
        'model_checkpoint-12345'

        >>> get_model_name_from_path("/path/to/model/final_model")
        'final_model'
    """
    # 去除路径两端的斜杠
    model_path = model_path.strip("/")
    # 按斜杠分割路径
    model_paths = model_path.split("/")
    # 检查路径的最后一部分是否以 "checkpoint-" 开头
    if model_paths[-1].startswith("checkpoint-"):
        # 返回 "上一级目录名_最后一部分"
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        # 返回路径的最后一部分
        return model_paths[-1]
