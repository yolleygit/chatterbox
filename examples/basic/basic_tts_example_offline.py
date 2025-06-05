#!/usr/bin/env python3
"""
Chatterbox TTS 基本使用示例 (离线版本)
演示最简单的文本转语音功能，使用本地缓存避免网络问题
"""

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import os
from pathlib import Path

def find_local_model_path():
    """查找本地缓存的模型路径"""
    # Hugging Face缓存路径
    hf_cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    # 查找chatterbox相关的目录
    for model_dir in hf_cache_dir.glob("models--ResembleAI--chatterbox*"):
        snapshots_dir = model_dir / "snapshots"
        if snapshots_dir.exists():
            # 获取最新的快照
            snapshots = list(snapshots_dir.iterdir())
            if snapshots:
                latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)
                return latest_snapshot
    
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

def basic_tts_demo_offline():
    """基本TTS演示（离线版本）"""
    print("🎤 初始化Chatterbox TTS模型（离线模式）...")
    
    # 应用设备映射补丁
    apply_torch_load_patch()
    
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

    # 首先尝试查找本地缓存
    local_path = find_local_model_path()
    
    try:
        if local_path and local_path.exists():
            print(f"📁 使用本地缓存: {local_path}")
            # 使用本地路径加载
            model = ChatterboxTTS.from_local(local_path, device="cpu")
        else:
            print("⏳ 本地缓存未找到，尝试在线下载...")
            model = ChatterboxTTS.from_pretrained(device="cpu")
        
        # 如果目标设备是MPS，需要特殊处理
        if device == "mps":
            print("🔄 将模型转移到MPS设备...")
            model = model.to(device)
            # 清理MPS缓存（如果可用）
            try:
                torch.mps.empty_cache()
            except AttributeError:
                pass
        elif device != "cpu":
            print(f"🔄 将模型转移到{device}设备...")
            model = model.to(device)
            
        print("✅ 模型加载完成")
        
    except Exception as e:
        print(f"⚠️  模型加载遇到问题: {e}")
        print("🔄 回退到CPU模式...")
        try:
            model = ChatterboxTTS.from_pretrained(device="cpu")
            device = "cpu"
            print("✅ CPU模式加载成功")
        except Exception as fallback_error:
            print(f"❌ CPU模式也失败了: {fallback_error}")
            print("💡 建议:")
            print("   1. 检查网络连接")
            print("   2. 尝试重启Python会话")
            print("   3. 确认conda环境正确激活")
            return

    # 要合成的文本
    text = "你好，我是Chatterbox，一个开源的语音合成系统。今天天气真不错！"
    
    print(f"📝 合成文本: {text}")
    print("⏳ 正在生成语音...")
    
    try:
        # 创建输出目录
        output_dir = Path("output/audio")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成语音
        with torch.no_grad():  # 减少内存使用
            wav = model.generate(text)
        
        # 保存音频文件
        output_path = output_dir / "basic_output_offline.wav"
        ta.save(output_path, wav, model.sr)
        
        print(f"🎵 语音已保存到: {output_path}")
        print(f"📊 采样率: {model.sr}Hz")
        print(f"⏱️  时长: {wav.shape[1] / model.sr:.2f}秒")
        print(f"🎯 使用设备: {device}")
        
        # 清理内存
        if device == "mps":
            try:
                torch.mps.empty_cache()
            except AttributeError:
                pass
                
    except Exception as e:
        print(f"❌ 语音生成失败: {e}")
        print("💡 可能的解决方案:")
        print("   1. 重启Python会话")
        print("   2. 确保有足够的内存")
        print("   3. 尝试使用CPU模式")

if __name__ == "__main__":
    basic_tts_demo_offline() 