import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from light_kinfer.utils.logger import get_logger

@pytest.fixture
def logger():
    # 初始化一个测试用的日志记录器
    return get_logger("test_logger")

def test_debug_logging(logger, caplog):
    logger._logger.propagate = True # 将日志记录器的propagate设置为True，确保日志传播到根记录器中
    # 测试 debug 级别日志
    with caplog.at_level("DEBUG", logger="test_logger"):
        logger.debug("This is a debug message")
        assert "This is a debug message" in caplog.text
        assert "DEBUG" in caplog.text

def test_info_logging(logger, caplog):
    logger._logger.propagate = True
    # 测试 info 级别日志
    with caplog.at_level("INFO", logger="test_logger"):
        logger.info("This is an info message")
        assert "This is an info message" in caplog.text
        assert "INFO" in caplog.text

def test_warning_logging(logger, caplog):
    logger._logger.propagate = True
    # 测试 warning 级别日志
    with caplog.at_level("WARNING", logger="test_logger"):
        logger.warning("This is a warning message")
        assert "This is a warning message" in caplog.text
        assert "WARNING" in caplog.text

def test_error_logging(logger, caplog):
    logger._logger.propagate = True
    # 测试 error 级别日志
    with caplog.at_level("ERROR", logger="test_logger"):
        logger.error("This is an error message")
        assert "This is an error message" in caplog.text
        assert "ERROR" in caplog.text

def test_critical_logging(logger, caplog):
    logger._logger.propagate = True
    # 测试 critical 级别日志
    with caplog.at_level("CRITICAL", logger="test_logger"):
        logger.critical("This is a critical message")
        assert "This is a critical message" in caplog.text
        assert "CRITICAL" in caplog.text
