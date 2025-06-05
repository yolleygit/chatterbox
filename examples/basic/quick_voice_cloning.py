#!/usr/bin/env python3
"""
语音克隆快速入门
简单易用的语音克隆示例
"""

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import os
from pathlib import Path

def apply_torch_load_patch():
    """应用torch.load补丁"""
    original_load = torch.load
    
    def patched_load(f, map_location=None, **kwargs):
        if map_location is None and not torch.cuda.is_available():
            map_location = torch.device('cpu')
        try:
            return original_load(f, map_location=map_location, **kwargs)
        except RuntimeError as e:
            if "CUDA" in str(e):
                return original_load(f, map_location=torch.device('cpu'), **kwargs)
            raise e
    
    torch.load = patched_load

def quick_voice_clone():
    """快速语音克隆演示"""
    print("🎭 语音克隆快速入门")
    print("=" * 40)
    
    # 应用补丁
    apply_torch_load_patch()
    
    # 创建输出目录
    output_dir = Path("output/quick_cloning")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("⏳ 加载模型...")
    model = ChatterboxTTS.from_pretrained(device="cpu")
    print("✅ 模型加载完成")
    
    # 检查是否有参考音频文件
    reference_files = [
        "voice_sample.wav",
        "reference.wav", 
        "audio_sample.wav",
        "my_voice.wav"
    ]
    
    reference_audio = None
    for file in reference_files:
        if os.path.exists(file):
            reference_audio = file
            break
    
    if reference_audio:
        print(f"🎯 找到参考音频: {reference_audio}")
        
        # 要克隆的文本
        texts = [
            "Hello, this is my cloned voice speaking.",
            "Voice cloning technology is truly amazing.",
            "I hope you enjoy this demonstration."
        ]
        
        print(f"🎵 开始语音克隆...")
        
        for i, text in enumerate(texts, 1):
            print(f"📝 正在处理第{i}段: {text}")
            
            # 语音克隆的核心代码
            wav = model.generate(
                text=text,
                audio_prompt_path=reference_audio,  # 关键：参考音频
                exaggeration=0.7,   # 情感表达
                cfg_weight=0.8,     # 相似度控制
                temperature=0.7     # 随机性控制
            )
            
            # 保存结果
            output_path = output_dir / f"cloned_{i}.wav"
            ta.save(output_path, wav, model.sr)
            print(f"  ✅ 保存到: {output_path}")
        
        print(f"\n🎉 语音克隆完成！文件保存在: {output_dir}")
        
    else:
        print("❌ 未找到参考音频文件")
        print("\n💡 请准备一个音频文件并命名为以下任一名称：")
        for file in reference_files:
            print(f"  • {file}")
        
        print("\n📝 音频文件要求：")
        print("  • 格式：WAV, MP3等")
        print("  • 时长：3-10秒")
        print("  • 质量：清晰无噪音")
        print("  • 内容：自然说话")
        
        print("\n🎙️ 录制方法：")
        print("  1. 打开Mac的语音备忘录或QuickTime")
        print("  2. 录制3-10秒自然说话")
        print("  3. 保存为voice_sample.wav")
        print("  4. 放在项目根目录")
        print("  5. 重新运行此脚本")

if __name__ == "__main__":
    quick_voice_clone() 