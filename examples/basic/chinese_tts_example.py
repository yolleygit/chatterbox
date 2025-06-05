#!/usr/bin/env python3
"""
中文语音合成示例
解决中文文本输出英文语音的问题
"""

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import os
from pathlib import Path
import time

def detect_device():
    """智能设备检测"""
    if torch.backends.mps.is_available():
        try:
            # 测试MPS设备是否正常工作
            test_tensor = torch.tensor([1.0], device='mps')
            _ = test_tensor + 1
            return 'mps'
        except:
            return 'cpu'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def create_chinese_voice_prompt():
    """
    创建中文语音提示
    注意：这里我们需要使用一个中文语音文件作为参考
    如果没有中文语音文件，Chatterbox会使用内置的英文语音模板
    """
    # 检查是否有中文语音文件
    chinese_voice_files = [
        "chinese_voice.wav",
        "中文语音.wav", 
        "output/chinese_reference.wav"
    ]
    
    for voice_file in chinese_voice_files:
        if os.path.exists(voice_file):
            print(f"✅ 找到中文语音文件: {voice_file}")
            return voice_file
    
    print("⚠️  未找到中文语音文件，将使用内置语音模板")
    print("💡 建议：")
    print("   1. 录制一段3-10秒的中文语音，保存为 'chinese_voice.wav'")
    print("   2. 或从网上下载中文语音样本")
    print("   3. 放在项目根目录下")
    return None

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

def main():
    print("🎤 中文语音合成示例")
    print("=" * 50)
    
    # 应用设备映射补丁
    apply_torch_load_patch()
    
    # 创建输出目录
    output_dir = Path("output/audio")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设备检测
    device = detect_device()
    print(f"🎯 使用设备: {device}")
    if device == "mps":
        print("🍎 启用Apple Metal加速")
    
    # 环境优化
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    if torch.get_num_threads() > 4:
        torch.set_num_threads(4)
    
    print("\n⏳ 正在加载Chatterbox TTS模型...")
    start_time = time.time()
    
    try:
        # 为Mac设备优化加载
        if device == "mps":
            model = ChatterboxTTS.from_pretrained(device="cpu")
            # 注意：Chatterbox TTS模型本身不支持.to()方法
            print("⚠️  模型已加载到CPU（Chatterbox TTS限制）")
            device_actual = "cpu"
        else:
            model = ChatterboxTTS.from_pretrained(device=device)
            device_actual = device
            
        load_time = time.time() - start_time
        print(f"✅ 模型加载完成 (耗时: {load_time:.2f}秒)")
        
    except RuntimeError as e:
        if "CUDA" in str(e):
            print("⚠️  检测到CUDA错误，回退到CPU模式...")
            model = ChatterboxTTS.from_pretrained(device="cpu")
            device_actual = "cpu"
            print("✅ CPU模式加载成功")
        else:
            raise e
    
    # 检查中文语音提示
    chinese_voice_prompt = create_chinese_voice_prompt()
    
    # 中文测试文本
    chinese_texts = [
        "你好，这是中文语音合成测试。",
        "今天天气很好，适合出去走走。",
        "人工智能技术发展非常迅速。"
    ]
    
    print(f"\n🎵 开始中文语音合成...")
    print(f"📁 输出目录: {output_dir}")
    
    for i, text in enumerate(chinese_texts, 1):
        print(f"\n📝 第{i}段: {text}")
        
        try:
            start_time = time.time()
            
            # 关键：使用中文语音提示
            if chinese_voice_prompt:
                wav_tensor = model.generate(
                    text=text,
                    audio_prompt_path=chinese_voice_prompt,  # 中文语音参考
                    exaggeration=0.8,  # 稍微增加情感表达
                    cfg_weight=0.6,    # 增加一致性
                    temperature=0.7    # 适中的随机性
                )
                print("  🎯 使用中文语音提示")
            else:
                # 没有中文语音提示时的设置
                wav_tensor = model.generate(
                    text=text,
                    exaggeration=0.5,  # 标准情感表达
                    cfg_weight=0.7,    # 高一致性，减少英文倾向
                    temperature=0.6    # 较低随机性
                )
                print("  ⚠️  使用内置语音模板（可能输出英文）")
            
            # 保存音频
            output_path = output_dir / f"chinese_output_{i}.wav"
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
    
    print(f"\n🎉 中文语音合成完成！")
    print(f"📂 所有文件保存在: {output_dir}")
    
    # 重要提示
    print(f"\n💡 重要说明:")
    print("1. 🎯 Chatterbox TTS主要为英文优化")
    print("2. 🗣️  要获得真正的中文语音，需要中文语音文件作为参考")
    print("3. 📁 请录制或下载中文语音样本，命名为 'chinese_voice.wav'")
    print("4. 🔄 使用中文语音参考可以显著改善中文合成效果")
    
    if not chinese_voice_prompt:
        print(f"\n⚠️  当前输出可能仍是英文语音，因为:")
        print("   - 模型使用英文分词器处理中文文本")
        print("   - 没有中文语音参考来引导发音")
        print("   - 需要中文voice prompt来'教'模型说中文")

if __name__ == "__main__":
    main() 