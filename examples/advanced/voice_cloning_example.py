#!/usr/bin/env python3
"""
Chatterbox TTS 语音克隆示例
演示如何使用参考音频进行零样本语音克隆
"""

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

def voice_cloning_demo():
    """语音克隆演示"""
    print("🎭 语音克隆演示")
    
    # 自动检测设备
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    print(f"🔧 使用设备: {device}")
    
    # 加载模型
    model = ChatterboxTTS.from_pretrained(device=device)
    
    # 参考音频路径（需要您提供实际的音频文件）
    REFERENCE_AUDIO = "reference_speaker.wav"  # 替换为您的参考音频
    
    # 要合成的文本
    text = "这是使用语音克隆技术生成的语音，声音应该与参考音频相似。"
    
    print(f"📁 参考音频: {REFERENCE_AUDIO}")
    print(f"📝 合成文本: {text}")
    
    try:
        print("⏳ 正在进行语音克隆...")
        
        # 使用参考音频进行语音克隆
        wav = model.generate(
            text, 
            audio_prompt_path=REFERENCE_AUDIO,
            exaggeration=0.5,      # 情感强度
            cfg_weight=0.5,        # 一致性权重
            temperature=0.8        # 随机性控制
        )
        
        # 保存克隆的语音
        output_path = "cloned_voice.wav"
        ta.save(output_path, wav, model.sr)
        
        print(f"🎵 克隆语音已保存到: {output_path}")
        
    except FileNotFoundError:
        print(f"❌ 找不到参考音频文件: {REFERENCE_AUDIO}")
        print("💡 请准备一个清晰的参考音频文件（建议3-10秒）")
    except Exception as e:
        print(f"❌ 语音克隆失败: {e}")

def create_reference_audio_guide():
    """参考音频准备指南"""
    print("\n📋 参考音频准备指南:")
    print("• 时长: 3-10秒为最佳")
    print("• 格式: WAV、MP3等常见格式")
    print("• 质量: 清晰，无背景噪音")
    print("• 内容: 自然说话，避免唱歌或特殊音效")
    print("• 采样率: 建议16kHz或更高")

if __name__ == "__main__":
    voice_cloning_demo()
    create_reference_audio_guide() 