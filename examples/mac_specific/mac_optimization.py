#!/usr/bin/env python3
"""
Mac M系列芯片优化配置
针对Apple Silicon进行的特殊优化
"""

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

def setup_mac_optimization():
    """设置Mac优化环境"""
    print("🍎 Mac M系列芯片优化设置")
    
    # 检测MPS支持
    if not torch.backends.mps.is_available():
        print("❌ MPS不可用，请检查macOS版本（需要12.3+）")
        return False
    
    print("✅ MPS可用，启用Apple Metal加速")
    
    # 设置设备
    device = "mps"
    map_location = torch.device(device)
    
    # 修补torch.load函数以确保正确的设备映射
    torch_load_original = torch.load
    def patched_torch_load(*args, **kwargs):
        if 'map_location' not in kwargs:
            kwargs['map_location'] = map_location
        return torch_load_original(*args, **kwargs)
    
    torch.load = patched_torch_load
    print("🔧 已应用设备映射补丁")
    
    return True

def mac_tts_demo():
    """Mac优化的TTS演示"""
    if not setup_mac_optimization():
        print("❌ Mac优化设置失败")
        return
    
    print("⏳ 加载模型（Mac优化版本）...")
    model = ChatterboxTTS.from_pretrained(device="mps")
    
    # 测试文本
    text = "这是在Mac M系列芯片上运行的Chatterbox TTS演示。Apple Silicon的强大性能让语音合成变得飞快！"
    
    print(f"📝 合成文本: {text}")
    print("🚀 使用Apple Metal加速生成...")
    
    try:
        # 生成语音（Mac优化参数）
        wav = model.generate(
            text,
            exaggeration=0.6,
            cfg_weight=0.5,
            temperature=0.8
        )
        
        # 保存音频
        output_path = "mac_optimized_output.wav"
        ta.save(output_path, wav, model.sr)
        
        duration = wav.shape[1] / model.sr
        print(f"✅ 生成完成！文件: {output_path}")
        print(f"⏱️  时长: {duration:.2f}秒")
        print(f"🔊 采样率: {model.sr}Hz")
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")

def mac_performance_tips():
    """Mac性能优化建议"""
    print("\n🚀 Mac性能优化建议:")
    print()
    print("💻 硬件配置:")
    print("• 推荐16GB+内存以获得最佳性能")
    print("• 确保有足够的存储空间（模型缓存）")
    print("• 关闭不必要的后台应用释放内存")
    print()
    print("⚙️ 系统设置:")
    print("• macOS 12.3+ 以支持MPS")
    print("• 启用高性能模式（系统偏好设置 > 电池）")
    print("• 确保良好的散热（避免过热降频）")
    print()
    print("🔧 环境配置:")
    print("• 使用conda环境隔离依赖")
    print("• 安装最新版本的PyTorch")
    print("• 设置合适的批处理大小")

if __name__ == "__main__":
    mac_tts_demo()
    mac_performance_tips() 