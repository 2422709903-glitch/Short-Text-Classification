
import torch
import sys
import os

print("=" * 50)
print("Python版本：", sys.version)
print("Python路径：", sys.executable)
print("PyTorch版本：", torch.__version__)
print("PyTorch安装路径：", torch.__file__)

print("=" * 50)
print("XPU是否可用：", torch.xpu.is_available())

if torch.xpu.is_available():

    print("XPU设备数量：", torch.xpu.device_count())

    for i in range(torch.xpu.device_count()):
        print(f"设备{i}：", torch.xpu.get_device_name(i))

    print("=" * 50)

    # 测试普通XPU张量计算
    try:
        a = torch.randn(100, 100, device="xpu")
        b = torch.randn(100, 100, device="xpu")

        c = torch.matmul(a, b)

        torch.xpu.synchronize()

        print("普通XPU矩阵计算：成功")
        print("结果设备：", c.device)

    except Exception as e:
        print("普通XPU矩阵计算：失败")
        print(repr(e))

print("=" * 50)

