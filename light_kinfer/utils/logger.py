import logging

class ColoredFormatter(logging.Formatter):
    """
    一个日志格式化器，根据日志级别输出带颜色的日志信息。
    """

    COLORS = {
        "DEBUG": "\033[36m",  # 蓝色
        "INFO": "\033[32m",  # 绿色
        "WARNING": "\033[33m",  # 黄色
        "ERROR": "\033[31m",  # 红色
        "CRITICAL": "\033[41m",  # 红色背景
    }
    RESET = "\033[0m"  # 重置颜色

    def format(self, record):
        """
        格式化指定的日志记录为文本。
        """
        # 获取与日志级别对应的颜色，如果没有则使用重置颜色
        color = self.COLORS.get(record.levelname, self.RESET)
        # 调用父类的方法格式化日志信息
        message = super().format(record)

        return f"{color}{message}{self.RESET}"

class SmartLogger:
    """
    日志记录器的包装类，提供增强的格式化支持。
    """
    
    def __init__(self, logger):
        self._logger = logger
    
    def debug(self, msg, *args, **kwargs):
        """
        输出调试级别的日志信息。
        """
        if self._logger.isEnabledFor(logging.DEBUG):
            self._logger.debug(msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        """
        输出信息级别的日志信息。
        """
        if self._logger.isEnabledFor(logging.INFO):
            self._logger.info(msg, *args, **kwargs)
    
    def warning(self, msg, *args, **kwargs):
        """
        输出警告级别的日志信息。
        """
        if self._logger.isEnabledFor(logging.WARNING):
            self._logger.warning(msg, *args, **kwargs)
    
    def error(self, msg, *args, **kwargs):
        """
        输出错误级别的日志信息。
        """
        if self._logger.isEnabledFor(logging.ERROR):
            self._logger.error(msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        """
        输出严重错误级别的日志信息。
        """
        if self._logger.isEnabledFor(logging.CRITICAL):
            self._logger.critical(msg, *args, **kwargs)

def get_logger(name):
    """
    获取一个带有颜色格式化的日志记录器。

    参数：
        name (str): 日志记录器的名称。

    返回：
        SmartLogger: 包装后的日志记录器。
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 设置日志级别为DEBUG
    
    # 防止日志重复记录，不向父记录器传播
    logger.propagate = False

    # 避免重复添加处理器
    if not logger.handlers:
        handler = logging.StreamHandler()  # 创建一个流处理器
        formatter = ColoredFormatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(filename)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S")  # 设置日志格式
        handler.setFormatter(formatter)  # 为处理器设置格式化器
        logger.addHandler(handler)  # 将处理器添加到记录器

    return SmartLogger(logger)  # 返回包装后的日志记录器
