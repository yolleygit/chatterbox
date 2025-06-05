#!/usr/bin/env python3
"""
MPS设备测试脚本
检查Mac上的MPS是否可用和工作正常
"""

import torch
import sys

def test_mps_availability():
    """测试MPS可用性"""
    print("🧪 测试MPS设备可用性...")
    print("=" * 50)
    
    # 检查MPS是否可用
    print(f"📋 PyTorch版本: {torch.__version__}")
    print(f"🍎 MPS可用: {torch.backends.mps.is_available()}")
    print(f"🏗️ MPS已构建: {torch.backends.mps.is_built()}")
    
    if torch.backends.mps.is_available():
        print("✅ MPS设备可用！")
        
        try:
            # 创建测试张量
            print("\n🧪 测试MPS张量运算...")
            device = torch.device("mps")
            x = torch.randn(5, 3, device=device)
            y = torch.randn(3, 4, device=device)
            z = torch.mm(x, y)
            print(f"✅ MPS张量运算成功: {z.shape}")
            
            # 测试CPU到MPS的转移
            print("\n🔄 测试设备转移...")
            cpu_tensor = torch.randn(2, 2)
            mps_tensor = cpu_tensor.to("mps")
            back_to_cpu = mps_tensor.to("cpu")
            print("✅ 设备转移测试成功")
            
            print("\n🎉 MPS设备完全可用，推荐使用！")
            return True
            
        except Exception as e:
            print(f"❌ MPS测试失败: {e}")
            print("💡 建议回退到CPU模式")
            return False
    else:
        print("❌ MPS设备不可用")
        if torch.cuda.is_available():
            print("✅ CUDA设备可用，将使用CUDA")
        else:
            print("💻 将使用CPU模式")
        return False

def apply_torch_load_patch():
    """应用torch.load补丁，修复CUDA设备映射问题"""
    original_load = torch.load
    
    def patched_load(f, map_location=None, **kwargs):
        # 如果没有指定map_location，并且CUDA不可用，自动映射到CPU
        if map_location is None and not torch.cuda.is_available():
            map_location = torch.device('cpu')
        try:
            return original_load(f, map_location=map_location, **kwargs)
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"⚠️ 自动修复CUDA设备映射错误")
                return original_load(f, map_location=torch.device('cpu'), **kwargs)
            raise e
    
    torch.load = patched_load

def test_chatterbox_with_mps():
    """测试Chatterbox TTS在MPS上的运行"""
    print("\n🎭 测试Chatterbox TTS MPS兼容性...")
    print("=" * 50)
    
    try:
        # 先应用torch.load补丁
        print("🔧 应用torch.load补丁...")
        apply_torch_load_patch()
        
        from chatterbox.tts import ChatterboxTTS
        
        if torch.backends.mps.is_available():
            print("🔄 尝试在MPS设备上加载模型...")
            model = ChatterboxTTS.from_pretrained(device="mps")
            print("✅ Chatterbox TTS在MPS上加载成功！")
            
            # 简单测试生成
            print("🧪 测试简单文本生成...")
            wav = model.generate("Hello, this is a test.")
            print(f"✅ 生成成功！音频长度: {wav.shape[1] / model.sr:.2f}秒")
            
            return True
        else:
            print("❌ MPS不可用，跳过测试")
            return False
            
    except Exception as e:
        print(f"❌ Chatterbox TTS MPS测试失败: {e}")
        print("💡 尝试CPU模式...")
        
        # 尝试CPU模式
        try:
            print("🔄 测试CPU模式...")
            model = ChatterboxTTS.from_pretrained(device="cpu")
            print("✅ Chatterbox TTS在CPU上加载成功！")
            return False  # MPS失败但CPU成功
        except Exception as cpu_e:
            print(f"❌ CPU模式也失败: {cpu_e}")
            return False

def main():
    """主测试函数"""
    print("🚀 Mac MPS设备兼容性测试")
    print("=" * 60)
    
    # 基本MPS测试
    mps_works = test_mps_availability()
    
    if mps_works:
        # Chatterbox测试
        chatterbox_works = test_chatterbox_with_mps()
        
        print("\n📊 测试总结:")
        print("=" * 30)
        print(f"🍎 MPS基础功能: {'✅ 可用' if mps_works else '❌ 不可用'}")
        print(f"🎭 Chatterbox MPS: {'✅ 兼容' if chatterbox_works else '❌ 不兼容'}")
        
        if mps_works and chatterbox_works:
            print("\n🎉 推荐设置:")
            print("   device='mps' - 使用Apple Silicon GPU加速")
        else:
            print("\n💡 推荐设置:")
            print("   device='cpu' - 使用CPU模式（更稳定）")
    else:
        print("\n💡 推荐设置:")
        print("   device='cpu' - 使用CPU模式")

if __name__ == "__main__":
    main() 