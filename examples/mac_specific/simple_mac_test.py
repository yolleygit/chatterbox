#!/usr/bin/env python3
"""
简化的Mac系统测试脚本
用于快速验证Chatterbox TTS的基本功能
"""

import torch
import os
import sys

def simple_mac_test():
    """简化的Mac兼容性测试"""
    print("🧪 简化Mac兼容性测试")
    print("=" * 40)
    
    # 1. 检测设备
    print("1. 🔍 检测计算设备...")
    if torch.backends.mps.is_available():
        device = "mps"
        print("   ✅ Apple Metal (MPS) 可用")
    elif torch.cuda.is_available():
        device = "cuda"
        print("   ✅ CUDA 可用")
    else:
        device = "cpu"
        print("   ✅ CPU 可用")
    
    print(f"   🎯 推荐设备: {device}")
    
    # 2. 设置环境优化
    print("\n2. ⚙️  应用环境优化...")
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    if torch.get_num_threads() > 4:
        torch.set_num_threads(4)
    print("   ✅ 环境优化完成")
    
    # 3. 测试PyTorch基本功能
    print("\n3. 🔧 测试PyTorch基本功能...")
    try:
        if device == "mps":
            test_tensor = torch.tensor([1.0, 2.0], device=device)
            result = test_tensor + 1
            print(f"   ✅ MPS张量运算: {result}")
        else:
            test_tensor = torch.tensor([1.0, 2.0])
            result = test_tensor + 1
            print(f"   ✅ CPU张量运算: {result}")
    except Exception as e:
        print(f"   ❌ 张量运算失败: {e}")
        return False
    
    # 4. 测试模型加载（使用补丁）
    print("\n4. 📦 测试模型加载补丁...")
    try:
        # 应用torch.load补丁
        original_load = torch.load
        
        def patched_load(f, map_location=None, **kwargs):
            if map_location is None and not torch.cuda.is_available():
                map_location = torch.device('cpu')
            try:
                return original_load(f, map_location=map_location, **kwargs)
            except RuntimeError as e:
                if "CUDA" in str(e):
                    print(f"      ⚠️  自动修复CUDA错误")
                    return original_load(f, map_location=torch.device('cpu'), **kwargs)
                raise e
        
        torch.load = patched_load
        print("   ✅ 模型加载补丁已应用")
        
    except Exception as e:
        print(f"   ❌ 补丁应用失败: {e}")
    
    # 5. 尝试加载Chatterbox TTS
    print("\n5. 🎤 测试Chatterbox TTS导入...")
    try:
        from chatterbox.tts import ChatterboxTTS
        print("   ✅ ChatterboxTTS模块导入成功")
        
        # 尝试模型初始化（使用CPU避免问题）
        print("   ⏳ 尝试模型初始化...")
        model = ChatterboxTTS.from_pretrained(device="cpu")
        print("   ✅ 模型初始化成功")
        
        # 简单文本测试
        print("   ⏳ 测试语音生成...")
        wav = model.generate("测试")
        print(f"   ✅ 语音生成成功 - 形状: {wav.shape}")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ ChatterboxTTS导入失败: {e}")
        print("   💡 请确保已安装: pip install chatterbox-tts")
        return False
    except Exception as e:
        print(f"   ❌ 模型测试失败: {e}")
        print(f"   📝 错误详情: {type(e).__name__}")
        return False

def main():
    """主函数"""
    print("🍎 Mac系统Chatterbox TTS简化测试")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print("")
    
    success = simple_mac_test()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 Mac兼容性测试通过！")
        print("✅ 可以安全使用Chatterbox TTS")
        print("\n📋 下一步:")
        print("   - 运行 python basic_tts_example.py")
        print("   - 或运行 python mac_tts_example.py")
    else:
        print("❌ Mac兼容性测试失败")
        print("\n🔧 建议:")
        print("   - 检查conda环境是否正确激活")
        print("   - 确认已安装chatterbox-tts")
        print("   - 尝试重启Python会话")

if __name__ == "__main__":
    main() 