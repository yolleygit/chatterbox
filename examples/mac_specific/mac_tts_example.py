#!/usr/bin/env python3
"""
Mac专用Chatterbox TTS示例
针对Mac系统（特别是Apple Silicon）进行优化
"""

# 首先应用Mac系统补丁
try:
    from mac_device_patch import apply_mac_optimizations, get_optimal_device
    apply_mac_optimizations()
except ImportError as e:
    print(f"⚠️  无法导入Mac补丁: {e}")
    print("⏩ 使用内置设备检测...")
    
    def get_optimal_device():
        """内置设备检测函数"""
        if torch.backends.mps.is_available():
            try:
                test_tensor = torch.tensor([1.0], device='mps')
                _ = test_tensor + 1
                return 'mps'
            except:
                return 'cpu'
        else:
            return 'cpu'

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import time

def mac_optimized_tts_demo():
    """Mac优化的TTS演示"""
    print("🍎 Mac专用Chatterbox TTS演示")
    print("=" * 50)
    
    # 获取最佳设备
    device = get_optimal_device()
    print(f"🎯 使用设备: {device}")
    
    # 显示系统信息
    if device == "mps":
        print("🚀 启用Apple Metal Performance Shaders加速")
    else:
        print("💻 使用CPU推理（适合所有Mac系统）")
    
    print("\n⏳ 正在加载模型...")
    start_time = time.time()
    
    try:
        # 先加载到CPU，再根据需要转移
        model = ChatterboxTTS.from_pretrained(device="cpu")
        
        if device == "mps":
            print("🔄 将模型转移到MPS设备...")
            # 逐步转移模型到MPS，避免内存问题
            model = model.to(device)
            try:
                torch.mps.empty_cache()  # 清理MPS缓存
            except AttributeError:
                # 某些PyTorch版本可能没有此功能
                pass
        
        load_time = time.time() - start_time
        print(f"✅ 模型加载完成 (耗时: {load_time:.2f}秒)")
        
    except Exception as e:
        print(f"⚠️  模型加载遇到问题: {e}")
        print("🔄 回退到CPU模式...")
        model = ChatterboxTTS.from_pretrained(device="cpu")
        device = "cpu"
        print("✅ CPU模式加载成功")
    
    # 测试文本
    texts = [
        "你好，我是专为Mac系统优化的Chatterbox语音合成系统。",
        "今天天气真不错，适合在家里测试语音合成技术。",
        "Apple Silicon芯片为AI推理提供了强大的计算能力。"
    ]
    
    print(f"\n🎵 开始语音合成演示...")
    
    for i, text in enumerate(texts, 1):
        print(f"\n📝 第{i}段: {text}")
        
        try:
            # 记录生成时间
            gen_start = time.time()
            
            # 生成语音
            with torch.no_grad():  # 减少内存使用
                wav = model.generate(text)
            
            gen_time = time.time() - gen_start
            
            # 保存音频
            output_path = f"mac_output_{i}.wav"
            ta.save(output_path, wav, model.sr)
            
            # 显示统计信息
            duration = wav.shape[1] / model.sr
            rtf = gen_time / duration  # 实时因子
            
            print(f"  ✅ 生成完成")
            print(f"  📁 保存路径: {output_path}")
            print(f"  ⏱️  生成耗时: {gen_time:.2f}秒")
            print(f"  🎼 音频时长: {duration:.2f}秒")
            print(f"  📊 实时因子: {rtf:.2f}x")
            
            # MPS设备内存管理
            if device == "mps":
                try:
                    torch.mps.empty_cache()
                except AttributeError:
                    # 某些PyTorch版本可能没有此功能
                    pass
                
        except Exception as e:
            print(f"  ❌ 生成失败: {e}")
            continue
    
    print(f"\n🎉 演示完成！")
    print(f"📊 总体性能 - 设备: {device}")
    
    # 提供Mac使用建议
    print(f"\n💡 Mac使用建议:")
    if device == "mps":
        print("  - Apple Silicon Mac可享受硬件加速")
        print("  - 建议关闭其他大型应用以释放内存")
        print("  - 如遇到内存不足，可重启Python会话")
    else:
        print("  - CPU模式虽然较慢但更稳定")
        print("  - 可以在后台运行其他轻量级任务")
    
    print("  - 定期保存重要音频文件")
    print("  - 推荐使用16GB或更大内存的Mac")

def quick_mac_test():
    """快速Mac兼容性测试"""
    print("🧪 快速Mac兼容性测试")
    
    try:
        device = get_optimal_device()
        print(f"✅ 设备检测: {device}")
        
        # 简单模型加载测试
        model = ChatterboxTTS.from_pretrained(device="cpu")
        print("✅ 模型加载: 成功")
        
        # 简单生成测试
        wav = model.generate("测试")
        print("✅ 语音生成: 成功")
        
        print("🎉 Mac兼容性测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 兼容性测试失败: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # 快速测试模式
        quick_mac_test()
    else:
        # 完整演示模式
        mac_optimized_tts_demo() 