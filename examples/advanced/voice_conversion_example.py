#!/usr/bin/env python3
"""
Chatterbox 语音转换示例
演示语音到语音的转换功能
"""

import torch
import torchaudio as ta
from chatterbox.vc import ChatterboxVC

def voice_conversion_demo():
    """语音转换演示"""
    print("🔄 Chatterbox 语音转换演示")
    
    # 检测设备
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🔧 使用设备: {device}")
    
    # 加载语音转换模型
    model = ChatterboxVC.from_pretrained(device)
    print("✅ 语音转换模型加载完成")
    
    # 输入音频和目标音频
    input_audio = "source_audio.wav"      # 要转换的音频
    target_voice = "target_speaker.wav"   # 目标说话人音频
    
    print(f"📁 源音频: {input_audio}")
    print(f"🎯 目标音频: {target_voice}")
    
    try:
        print("⏳ 正在进行语音转换...")
        
        # 执行语音转换
        converted_wav = model.generate(
            audio=input_audio,
            target_voice_path=target_voice
        )
        
        # 保存转换结果
        output_path = "converted_voice.wav"
        ta.save(output_path, converted_wav, model.sr)
        
        duration = converted_wav.shape[1] / model.sr
        print(f"🎵 转换完成！已保存到: {output_path}")
        print(f"⏱️  时长: {duration:.2f}秒")
        
    except FileNotFoundError as e:
        print(f"❌ 找不到音频文件: {e}")
        print("💡 请确保源音频和目标音频文件存在")
    except Exception as e:
        print(f"❌ 语音转换失败: {e}")

def voice_conversion_tips():
    """语音转换使用技巧"""
    print("\n💡 语音转换使用技巧:")
    print()
    print("📋 音频质量要求:")
    print("• 清晰度: 高质量录音，无噪音")
    print("• 时长: 源音频任意长度，目标音频3-10秒")
    print("• 内容: 目标音频应包含自然说话")
    print("• 格式: 支持WAV、MP3等常见格式")
    print()
    print("🎯 最佳实践:")
    print("• 目标说话人音频越清晰，转换效果越好")
    print("• 避免背景音乐或噪音")
    print("• 语速相近的音频转换效果更佳")
    print("• 可尝试不同的目标音频片段")

if __name__ == "__main__":
    voice_conversion_demo()
    voice_conversion_tips() 