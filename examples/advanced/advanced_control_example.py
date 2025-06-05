#!/usr/bin/env python3
"""
Chatterbox TTS 高级控制示例
演示情感夸张控制、CFG权重调节等高级功能
"""

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

def advanced_control_demo():
    """高级参数控制演示"""
    print("🎛️ Chatterbox TTS 高级参数控制演示")
    
    # 检测设备
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🔧 使用设备: {device}")
    
    # 加载模型
    model = ChatterboxTTS.from_pretrained(device=device)
    
    # 测试文本
    text = "今天真是美好的一天！阳光明媚，鸟儿在歌唱，一切都充满了希望和活力。"
    
    # 不同的参数配置
    configs = [
        {
            "name": "中性默认",
            "params": {"exaggeration": 0.5, "cfg_weight": 0.5, "temperature": 0.8},
            "file": "neutral_default.wav"
        },
        {
            "name": "极度夸张",
            "params": {"exaggeration": 2.0, "cfg_weight": 0.3, "temperature": 0.8},
            "file": "extreme_exaggeration.wav"
        },
        {
            "name": "平静稳重",
            "params": {"exaggeration": 0.25, "cfg_weight": 0.7, "temperature": 0.6},
            "file": "calm_stable.wav"
        },
        {
            "name": "自然随性",
            "params": {"exaggeration": 0.7, "cfg_weight": 0.3, "temperature": 1.0},
            "file": "natural_casual.wav"
        }
    ]
    
    print(f"📝 测试文本: {text}\n")
    
    for config in configs:
        print(f"🎯 生成配置: {config['name']}")
        print(f"   参数: {config['params']}")
        print("   ⏳ 生成中...")
        
        try:
            wav = model.generate(text, **config['params'])
            ta.save(config['file'], wav, model.sr)
            
            duration = wav.shape[1] / model.sr
            print(f"   ✅ 已保存: {config['file']} (时长: {duration:.2f}s)")
            
        except Exception as e:
            print(f"   ❌ 生成失败: {e}")
        
        print()

def parameter_guide():
    """参数调节指南"""
    print("📖 参数调节指南:")
    print()
    print("🎭 exaggeration (情感夸张度):")
    print("   • 0.25-0.5: 平静、正式的语调")
    print("   • 0.5-0.7: 自然、日常的表达")
    print("   • 0.7-1.5: 生动、有表现力")
    print("   • 1.5-2.0: 极度夸张、戏剧性")
    print()
    print("⚖️ cfg_weight (分类器自由引导权重):")
    print("   • 0.1-0.3: 更自然，但可能不稳定")
    print("   • 0.4-0.6: 平衡自然度和一致性")
    print("   • 0.7-0.9: 高一致性，但可能僵硬")
    print()
    print("🌡️ temperature (随机性控制):")
    print("   • 0.5-0.7: 保守、可预测")
    print("   • 0.8-1.0: 自然变化")
    print("   • 1.0+: 更多随机性和变化")

def usage_scenarios():
    """使用场景建议"""
    print("🎬 使用场景建议:")
    print()
    print("📢 新闻播报/正式场合:")
    print("   exaggeration=0.3, cfg_weight=0.7, temperature=0.6")
    print()
    print("🎭 有声小说/戏剧表演:")
    print("   exaggeration=1.2, cfg_weight=0.4, temperature=0.9")
    print()
    print("💬 日常对话/聊天机器人:")
    print("   exaggeration=0.6, cfg_weight=0.5, temperature=0.8")
    print()
    print("🎵 广告/营销内容:")
    print("   exaggeration=0.8, cfg_weight=0.4, temperature=0.7")

if __name__ == "__main__":
    advanced_control_demo()
    print("="*50)
    parameter_guide()
    print("="*50)
    usage_scenarios() 