#!/usr/bin/env python3
"""
Chatterbox TTS 基本使用示例
演示最简单的文本转语音功能
"""

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

def basic_tts_demo():
    """基本TTS演示"""
    print("🎤 初始化Chatterbox TTS模型...")
    
    # 自动检测最佳设备
    if torch.cuda.is_available():
        device = "cuda"
        print("🚀 使用CUDA加速")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("🍎 使用Apple Metal加速")
    else:
        device = "cpu"
        print("💻 使用CPU推理")

    # 加载预训练模型（首次运行会自动下载）
    # Mac系统需要特殊处理设备映射
    try:
        if device == "mps":
            # Apple Silicon Mac需要映射到CPU再转移到MPS
            model = ChatterboxTTS.from_pretrained(device="cpu")
            # 将模型移动到MPS设备
            model = model.to(device)
            print("✅ 模型已加载并映射到Apple Metal")
        else:
            model = ChatterboxTTS.from_pretrained(device=device)
            print("✅ 模型加载完成")
    except RuntimeError as e:
        if "CUDA" in str(e) and "torch.cuda.is_available() is False" in str(e):
            print("⚠️  检测到CUDA设备映射错误，使用CPU作为备选...")
            model = ChatterboxTTS.from_pretrained(device="cpu")
            print("✅ 模型已加载到CPU")
            device = "cpu"  # 更新设备状态
        else:
            raise e

    # 要合成的文本
    text = "你好，我是Chatterbox，一个开源的语音合成系统。今天天气真不错！"
    
    print(f"📝 合成文本: {text}")
    print("⏳ 正在生成语音...")
    
    # 生成语音
    wav = model.generate(text)
    
    # 保存音频文件
    output_path = "basic_output.wav"
    ta.save(output_path, wav, model.sr)
    
    print(f"🎵 语音已保存到: {output_path}")
    print(f"📊 采样率: {model.sr}Hz")
    print(f"⏱️  时长: {wav.shape[1] / model.sr:.2f}秒")

if __name__ == "__main__":
    basic_tts_demo() 