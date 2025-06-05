#!/usr/bin/env python3
"""
简化版语音克隆启动脚本
更清晰、更人性化的界面体验
"""

import os
import sys

def main():
    """启动简化版语音克隆网页界面"""
    
    print("\n" + "="*60)
    print("🎭 简化版智能语音克隆系统")
    print("="*60)
    
    print("\n🌟 界面特点:")
    print("  • 🎯 清晰的三步操作流程")
    print("  • 🎨 美观的卡片式布局") 
    print("  • 🚀 智能预设，简化参数")
    print("  • 📱 人性化的用户体验")
    print("  • 💾 历史录音快速重用")
    
    print("\n✨ 相比完整版的改进:")
    print("  • 减少了复杂的参数调整")
    print("  • 简化了音频库管理")
    print("  • 优化了页面布局和视觉效果")
    print("  • 提供了智能预设选项")
    print("  • 更清晰的步骤引导")
    
    print("\n🌐 访问地址:")
    print("  • 本地访问: http://localhost:7863")
    print("  • 局域网访问: http://0.0.0.0:7863")
    
    print("\n💡 使用提示:")
    print("  • 模型会自动加载，请等待初始化完成")
    print("  • 录制3-10秒清晰音频效果最佳")
    print("  • 可以保存录音到历史记录重复使用")
    print("  • 三种预设模式满足不同需求")
    
    print("\n⏳ 正在启动简化版服务器...")
    print("🔄 如需停止，请按 Ctrl+C")
    print()
    
    # 导入并启动界面
    try:
        from examples.gradio_demos.simple_final_voice_cloning_web import main as start_simple_interface
        start_simple_interface()
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("💡 请确保在项目根目录运行此脚本")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 用户中断，已停止服务器")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 