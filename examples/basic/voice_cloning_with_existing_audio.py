#!/usr/bin/env python3
"""
使用现有音频文件进行语音克隆演示
利用项目中已有的音频文件作为参考
"""

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import os
from pathlib import Path
import glob

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

def find_existing_audio_files():
    """查找项目中现有的音频文件"""
    audio_patterns = [
        "output/**/*.wav",
        "audio_output/*.wav", 
        "*.wav",
        "**/*.mp3",
        "**/*.flac"
    ]
    
    audio_files = []
    for pattern in audio_patterns:
        files = glob.glob(pattern, recursive=True)
        audio_files.extend(files)
    
    # 去重并过滤
    unique_files = list(set(audio_files))
    # 过滤掉可能的输出文件（避免使用克隆的结果作为参考）
    valid_files = [f for f in unique_files if 'cloned' not in f.lower() and 'voice_cloning' not in f.lower()]
    
    return valid_files

def analyze_audio_quality(audio_path):
    """分析音频文件基本信息"""
    try:
        # 使用torchaudio加载音频
        waveform, sample_rate = ta.load(audio_path)
        duration = waveform.shape[1] / sample_rate
        channels = waveform.shape[0]
        max_amplitude = torch.max(torch.abs(waveform)).item()
        
        print(f"  📊 文件分析:")
        print(f"     📁 路径: {audio_path}")
        print(f"     ⏱️  时长: {duration:.2f}秒")
        print(f"     📈 采样率: {sample_rate}Hz")
        print(f"     🔊 声道数: {channels}")
        print(f"     📈 最大振幅: {max_amplitude:.3f}")
        
        # 质量评估
        quality_score = 0
        issues = []
        
        if 2 <= duration <= 15:
            quality_score += 2
        elif duration < 2:
            issues.append("时长过短")
        elif duration > 15:
            issues.append("时长过长")
        else:
            quality_score += 1
            
        if sample_rate >= 16000:
            quality_score += 2
        else:
            issues.append("采样率偏低")
            
        if 0.1 <= max_amplitude <= 0.9:
            quality_score += 2
        elif max_amplitude < 0.1:
            issues.append("音量过小")
        elif max_amplitude > 0.95:
            issues.append("可能有削波失真")
        else:
            quality_score += 1
        
        if quality_score >= 5:
            print(f"     ✅ 质量评分: {quality_score}/6 (优秀)")
        elif quality_score >= 3:
            print(f"     🟡 质量评分: {quality_score}/6 (良好)")
        else:
            print(f"     🔴 质量评分: {quality_score}/6 (一般)")
            
        if issues:
            print(f"     ⚠️  注意事项: {', '.join(issues)}")
            
        return quality_score >= 3, duration
        
    except Exception as e:
        print(f"     ❌ 分析失败: {e}")
        return False, 0

def voice_cloning_demo_with_existing():
    """使用现有音频文件进行语音克隆演示"""
    print("🎭 使用现有音频文件进行语音克隆")
    print("=" * 50)
    
    # 应用补丁
    apply_torch_load_patch()
    
    # 创建输出目录
    output_dir = Path("output/voice_cloning")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔍 搜索现有音频文件...")
    audio_files = find_existing_audio_files()
    
    if not audio_files:
        print("❌ 未找到可用的音频文件")
        print("💡 请先运行基础TTS示例生成一些音频文件，或提供自己的音频文件")
        return
    
    print(f"✅ 找到 {len(audio_files)} 个音频文件")
    
    # 分析并选择最佳音频文件
    best_file = None
    best_score = 0
    
    print("\n📊 分析音频文件质量...")
    for i, audio_file in enumerate(audio_files[:5], 1):  # 最多分析5个文件
        print(f"\n{i}. 分析文件: {audio_file}")
        is_good, duration = analyze_audio_quality(audio_file)
        
        if is_good and duration > best_score:
            best_file = audio_file
            best_score = duration
    
    if not best_file:
        # 如果没有找到高质量文件，使用第一个
        best_file = audio_files[0]
        print(f"\n⚠️  未找到高质量音频，将使用: {best_file}")
    else:
        print(f"\n🎯 选择最佳音频文件: {best_file}")
    
    print("\n⏳ 加载Chatterbox TTS模型...")
    try:
        model = ChatterboxTTS.from_pretrained(device="cpu")
        print("✅ 模型加载完成")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 测试文本（针对语音克隆优化的内容）
    test_texts = [
        "This is a voice cloning demonstration using existing audio.",
        "The technology can reproduce voice characteristics remarkably well.",
        "Each generation may sound slightly different due to the random nature.",
        "Voice cloning opens up many creative possibilities."
    ]
    
    print(f"\n🎵 开始语音克隆演示...")
    print(f"📝 将生成 {len(test_texts)} 段语音")
    
    successful_generations = 0
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 第{i}段文本: {text}")
        
        try:
            # 语音克隆关键代码
            wav_tensor = model.generate(
                text=text,
                audio_prompt_path=best_file,  # 使用选中的音频文件作为参考
                exaggeration=0.7,             # 情感表达度
                cfg_weight=0.8,               # 相似度权重（高相似度）
                temperature=0.7               # 随机性控制
            )
            
            # 保存生成的音频
            output_path = output_dir / f"cloned_from_existing_{i}.wav"
            ta.save(output_path, wav_tensor, model.sr)
            
            # 计算音频时长
            audio_duration = wav_tensor.shape[1] / model.sr
            
            print(f"  ✅ 生成成功")
            print(f"  📁 保存路径: {output_path}")
            print(f"  🎼 音频时长: {audio_duration:.2f}秒")
            
            successful_generations += 1
            
        except Exception as e:
            print(f"  ❌ 生成失败: {e}")
            continue
    
    print(f"\n🎉 语音克隆演示完成！")
    print(f"📊 成功生成: {successful_generations}/{len(test_texts)} 个音频文件")
    print(f"📂 输出目录: {output_dir}")
    print(f"🎯 参考音频: {best_file}")
    
    # 提供进一步的建议
    print("\n💡 进一步优化建议:")
    print("1. 📎 尝试使用自己录制的高质量音频作为参考")
    print("2. 🎛️  调整参数: cfg_weight (相似度), exaggeration (表达度), temperature (随机性)")
    print("3. 📝 使用更长或更短的文本进行测试")
    print("4. 🔄 多次生成同一文本，观察变化")
    
    print("\n🎛️ 参数说明:")
    print("• cfg_weight=0.8 (高相似度) - 可调整到0.5-0.9")
    print("• exaggeration=0.7 (中等表达) - 可调整到0.3-1.2") 
    print("• temperature=0.7 (中等随机) - 可调整到0.5-1.0")
    
    print("\n📖 相关文档:")
    print("• docs/语音克隆完整指南.md - 详细使用指南")
    print("• examples/advanced/voice_cloning_tutorial.py - 高级教程")

if __name__ == "__main__":
    voice_cloning_demo_with_existing() 