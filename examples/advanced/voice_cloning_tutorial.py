#!/usr/bin/env python3
"""
语音克隆详细教程
演示如何使用Chatterbox TTS进行语音克隆
"""

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import os
from pathlib import Path
import time
import librosa
import numpy as np

def apply_torch_load_patch():
    """应用torch.load补丁，处理设备映射问题"""
    original_load = torch.load
    
    def patched_load(f, map_location=None, **kwargs):
        if map_location is None and not torch.cuda.is_available():
            map_location = torch.device('cpu')
        try:
            return original_load(f, map_location=map_location, **kwargs)
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"⚠️  自动修复CUDA设备映射错误")
                return original_load(f, map_location=torch.device('cpu'), **kwargs)
            raise e
    
    torch.load = patched_load

def detect_device():
    """智能设备检测"""
    if torch.backends.mps.is_available():
        try:
            test_tensor = torch.tensor([1.0], device='mps')
            _ = test_tensor + 1
            return 'mps'
        except:
            return 'cpu'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def analyze_audio_file(audio_path):
    """分析音频文件质量"""
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        duration = len(audio) / sr
        
        print(f"📊 音频分析结果:")
        print(f"   📁 文件: {audio_path}")
        print(f"   ⏱️  时长: {duration:.2f}秒")
        print(f"   📈 采样率: {sr}Hz")
        print(f"   🔊 最大音量: {np.max(np.abs(audio)):.3f}")
        print(f"   💾 文件大小: {os.path.getsize(audio_path)/1024:.1f}KB")
        
        # 质量建议
        if duration < 3:
            print(f"   ⚠️  时长偏短，建议3-10秒")
        elif duration > 20:
            print(f"   ⚠️  时长偏长，建议3-10秒")
        else:
            print(f"   ✅ 时长合适")
            
        if sr < 16000:
            print(f"   ⚠️  采样率偏低，建议16kHz+")
        else:
            print(f"   ✅ 采样率合适")
            
        return True
    except Exception as e:
        print(f"   ❌ 分析失败: {e}")
        return False

def find_voice_samples():
    """查找可用的语音样本文件"""
    voice_extensions = ['.wav', '.mp3', '.flac', '.m4a']
    voice_samples = []
    
    # 检查常见位置
    search_paths = [
        ".",  # 当前目录
        "voice_samples/",
        "audio_samples/", 
        "samples/",
        "voices/"
    ]
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            for file in os.listdir(search_path):
                if any(file.lower().endswith(ext) for ext in voice_extensions):
                    full_path = os.path.join(search_path, file)
                    voice_samples.append(full_path)
    
    return voice_samples

def voice_cloning_demo():
    """语音克隆演示"""
    print("🎭 语音克隆详细教程")
    print("=" * 60)
    
    # 应用补丁和环境设置
    apply_torch_load_patch()
    device = detect_device()
    print(f"🎯 使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path("output/voice_cloning")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 环境优化
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    if torch.get_num_threads() > 4:
        torch.set_num_threads(4)
    
    print("\n⏳ 正在加载Chatterbox TTS模型...")
    start_time = time.time()
    
    try:
        model = ChatterboxTTS.from_pretrained(device="cpu")
        load_time = time.time() - start_time
        print(f"✅ 模型加载完成 (耗时: {load_time:.2f}秒)")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    print("\n" + "="*60)
    print("📚 语音克隆教程")
    print("="*60)
    
    print("\n1️⃣ 什么是语音克隆？")
    print("语音克隆是通过提供一个参考音频文件，让AI模型学习其声音特征，")
    print("然后用这个声音特征来合成新的文本内容。")
    
    print("\n2️⃣ 语音克隆的工作原理：")
    print("参考音频 → 声音特征提取 → 特征融合 → 新文本合成 → 目标声音输出")
    
    print("\n3️⃣ 音频文件要求：")
    print("✅ 格式: WAV, MP3, FLAC等")
    print("✅ 时长: 3-10秒（推荐）")
    print("✅ 质量: 清晰、无噪音")
    print("✅ 内容: 自然说话，不要朗读")
    print("✅ 采样率: 16kHz或更高")
    
    print("\n4️⃣ 查找可用的音频文件...")
    voice_samples = find_voice_samples()
    
    if voice_samples:
        print(f"✅ 找到 {len(voice_samples)} 个音频文件:")
        for i, sample in enumerate(voice_samples[:5], 1):  # 只显示前5个
            print(f"   {i}. {sample}")
        
        # 使用第一个找到的音频文件进行演示
        reference_audio = voice_samples[0]
        print(f"\n🎯 选择音频文件进行演示: {reference_audio}")
        
        # 分析音频文件
        if analyze_audio_file(reference_audio):
            print("\n5️⃣ 开始语音克隆演示...")
            demo_voice_cloning(model, reference_audio, output_dir)
        else:
            print("❌ 音频文件分析失败，无法进行演示")
    else:
        print("❌ 未找到音频文件")
        print("\n💡 请按以下步骤准备音频文件：")
        create_sample_instructions()

def demo_voice_cloning(model, reference_audio, output_dir):
    """演示语音克隆过程"""
    print(f"\n🎭 使用 {reference_audio} 进行语音克隆...")
    
    # 测试文本
    test_texts = [
        "Hello, this is a voice cloning demonstration.",
        "The weather is beautiful today, perfect for a walk.",
        "Artificial intelligence technology is advancing rapidly.",
        "Thank you for trying out the voice cloning feature."
    ]
    
    print(f"📝 将合成 {len(test_texts)} 段文本")
    
    total_start = time.time()
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 第{i}段: {text}")
        
        try:
            start_time = time.time()
            
            # 关键：使用参考音频进行语音克隆
            wav_tensor = model.generate(
                text=text,
                audio_prompt_path=reference_audio,  # 参考音频文件
                exaggeration=0.6,  # 情感表达度
                cfg_weight=0.7,    # 一致性权重，越高越像参考音频
                temperature=0.8    # 随机性，影响声音变化
            )
            
            # 保存音频
            output_path = output_dir / f"cloned_voice_{i}.wav"
            ta.save(output_path, wav_tensor, model.sr)
            
            generation_time = time.time() - start_time
            audio_duration = wav_tensor.shape[1] / model.sr
            rtf = generation_time / audio_duration
            
            print(f"  ✅ 生成完成")
            print(f"  📁 保存路径: {output_path}")
            print(f"  ⏱️  生成耗时: {generation_time:.2f}秒")
            print(f"  🎼 音频时长: {audio_duration:.2f}秒")
            print(f"  📊 实时因子: {rtf:.2f}x")
            
        except Exception as e:
            print(f"  ❌ 生成失败: {e}")
            continue
    
    total_time = time.time() - total_start
    print(f"\n🎉 语音克隆演示完成！")
    print(f"📂 所有文件保存在: {output_dir}")
    print(f"⏱️  总耗时: {total_time:.2f}秒")
    
    # 参数调优建议
    print_parameter_tuning_tips()

def create_sample_instructions():
    """创建音频样本准备说明"""
    print("\n📖 音频样本准备指南:")
    print("="*40)
    
    print("\n🎙️ 录制方法:")
    print("1. 使用Mac内置录音应用")
    print("2. 或使用QuickTime Player → 文件 → 新建音频录制")
    print("3. 录制3-10秒自然说话")
    print("4. 保存为voice_sample.wav")
    
    print("\n💾 下载方法:")
    print("1. 从语音数据集下载（如CommonVoice）")
    print("2. 从播客或视频中提取音频片段")
    print("3. 使用其他TTS工具生成样本")
    
    print("\n📁 文件放置:")
    print("将音频文件放在以下任一位置：")
    print("- 项目根目录")
    print("- voice_samples/ 目录")
    print("- audio_samples/ 目录")
    
    print("\n🔧 音频处理工具推荐:")
    print("- Audacity (免费，跨平台)")
    print("- FFmpeg (命令行工具)")
    print("- Mac内置音频处理")

def print_parameter_tuning_tips():
    """打印参数调优建议"""
    print("\n🎛️ 参数调优指南:")
    print("="*40)
    
    print("\n📊 cfg_weight (一致性权重) 0.1-1.0:")
    print("  • 0.3-0.5: 更有创意，声音变化较大")
    print("  • 0.6-0.8: 平衡，推荐值")
    print("  • 0.8-1.0: 更像参考音频，更一致")
    
    print("\n🎭 exaggeration (情感表达) 0.1-2.0:")
    print("  • 0.1-0.4: 平淡，适合正式场合")
    print("  • 0.5-0.8: 自然，推荐值")
    print("  • 0.9-2.0: 夸张，适合戏剧表演")
    
    print("\n🎲 temperature (随机性) 0.1-2.0:")
    print("  • 0.1-0.5: 更稳定，重复性好")
    print("  • 0.6-1.0: 平衡，推荐值")
    print("  • 1.1-2.0: 更多样，每次不同")
    
    print("\n💡 推荐组合:")
    print("  • 高质量克隆: cfg_weight=0.8, exaggeration=0.6, temperature=0.7")
    print("  • 创意表达: cfg_weight=0.5, exaggeration=1.0, temperature=1.2")
    print("  • 稳定输出: cfg_weight=0.9, exaggeration=0.4, temperature=0.5")

def main():
    """主函数"""
    print("🚀 启动语音克隆教程...")
    voice_cloning_demo()
    
    print("\n" + "="*60)
    print("📚 补充说明")
    print("="*60)
    
    print("\n✨ 语音克隆的应用场景:")
    print("• 📖 有声书制作")
    print("• 🎬 视频配音")
    print("• 🤖 个性化语音助手")
    print("• 🎵 音乐和播客制作")
    print("• 🌍 多语言内容本地化")
    
    print("\n⚖️ 使用注意事项:")
    print("• 🔒 仅用于合法和道德目的")
    print("• 👤 获得声音所有者同意")
    print("• 🚫 不用于欺诈或误导")
    print("• 📜 遵守当地法律法规")
    
    print("\n🔗 相关文件:")
    print("• examples/advanced/voice_cloning_example.py - 基础语音克隆")
    print("• examples/basic/chinese_tts_example.py - 中文语音克隆")
    print("• 中文语音合成解决方案.md - 详细解决方案")

if __name__ == "__main__":
    main() 