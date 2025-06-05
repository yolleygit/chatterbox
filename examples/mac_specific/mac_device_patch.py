#!/usr/bin/env python3
"""
Mac设备映射补丁
解决在Mac系统上运行Chatterbox TTS时的CUDA设备映射问题
"""

import torch
import os
from pathlib import Path

def patch_torch_load():
    """
    为Mac系统修补torch.load函数，自动处理设备映射
    """
    original_load = torch.load
    
    def patched_load(f, map_location=None, **kwargs):
        """修补的torch.load函数，自动处理设备映射"""
        
        # 如果没有指定map_location且在Mac上，自动映射到CPU
        if map_location is None:
            if torch.backends.mps.is_available():
                # Apple Silicon Mac - 先加载到CPU
                map_location = torch.device('cpu')
            elif not torch.cuda.is_available():
                # 其他Mac系统 - 映射到CPU
                map_location = torch.device('cpu')
        
        try:
            return original_load(f, map_location=map_location, **kwargs)
        except RuntimeError as e:
            if "CUDA" in str(e) and "torch.cuda.is_available() is False" in str(e):
                print(f"⚠️  自动修复CUDA设备映射问题: {f}")
                return original_load(f, map_location=torch.device('cpu'), **kwargs)
            else:
                raise e
    
    # 替换torch.load
    torch.load = patched_load
    print("🔧 Mac设备映射补丁已应用")

def apply_mac_optimizations():
    """
    应用Mac系统的各种优化设置
    """
    # 应用torch.load补丁
    patch_torch_load()
    
    # 设置环境变量优化
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 启用MPS回退
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # 内存管理优化
    
    # 设置多线程优化
    if torch.get_num_threads() > 4:
        torch.set_num_threads(4)  # 限制线程数，避免过度占用
    
    print("🍎 Mac系统优化设置已应用")

def get_optimal_device():
    """
    获取Mac系统的最佳计算设备
    
    Returns:
        str: 最佳设备名称 ('mps', 'cpu')
    """
    if torch.backends.mps.is_available():
        # 检查MPS是否真正可用
        try:
            test_tensor = torch.tensor([1.0], device='mps')
            _ = test_tensor + 1
            return 'mps'
        except:
            print("⚠️  MPS设备检测失败，回退到CPU")
            return 'cpu'
    else:
        return 'cpu'

def verify_mac_setup():
    """
    验证Mac系统设置
    """
    print("🔍 验证Mac系统设置...")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"MPS可用: {torch.backends.mps.is_available()}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    device = get_optimal_device()
    print(f"推荐设备: {device}")
    
    if device == 'mps':
        print("🚀 您的Mac支持Metal Performance Shaders加速")
    else:
        print("💻 将使用CPU进行推理")
    
    return device

if __name__ == "__main__":
    # 应用Mac优化
    apply_mac_optimizations()
    
    # 验证设置
    device = verify_mac_setup()
    
    print("\n📋 使用建议:")
    print("1. 在导入chatterbox之前运行此补丁")
    print("2. 使用返回的device参数初始化模型") 
    print("3. 如遇到内存问题，重启Python会话") 