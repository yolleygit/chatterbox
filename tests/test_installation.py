#!/usr/bin/env python3
"""
Chatterbox TTS 安装验证脚本
测试所有核心模块是否正确安装
"""

def test_installation():
    """验证Chatterbox TTS安装是否正确"""
    print("🔍 正在验证Chatterbox TTS安装...")
    
    try:
        # 测试核心导入
        from chatterbox.tts import ChatterboxTTS
        from chatterbox.vc import ChatterboxVC
        print("✅ 核心模块导入成功")
        
        # 测试PyTorch和设备检测
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            print("✅ Apple Metal (MPS) 可用")
        else:
            print("⚠️  仅CPU可用，建议使用GPU加速")
            
        print("🎉 安装验证完成！")
        return True
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False

if __name__ == "__main__":
    test_installation() 