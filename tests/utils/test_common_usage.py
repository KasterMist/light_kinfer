import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from light_kinfer.utils.common import *

def test_get_gpu_memory_used():
    device = detect_device()
    if device == "nvidia":
        mem = get_gpu_memory(gpu_type="nvidia", device_id="0")
        print("NVIDIA GPU mem:", mem)
        assert mem is None or isinstance(mem, float)
    elif device == "amd":
        mem = get_gpu_memory(gpu_type="amd", device_id="0")
        print("AMD GPU mem:", mem)
        assert mem is None or isinstance(mem, float)
    else:
        assert device == "cpu"

def test_getProjectPath():
    path = getProjectPath()
    print("Project Path:", path)
    assert os.path.exists(path)