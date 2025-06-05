#!/usr/bin/env python3
"""
语音克隆网页界面启动脚本
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from examples.gradio_demos.voice_cloning_web import main

if __name__ == "__main__":
    print("🚀 启动语音克隆网页界面...")
    print("📱 功能特性:")
    print("  • 🎙️ 麦克风录音")
    print("  • 📁 文件上传")
    print("  • 🎛️ 参数调优")
    print("  • 🎭 实时克隆")
    print("  • 📊 音频分析")
    print("\n⏳ 正在启动服务器...")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 用户中断，关闭服务器")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        print("💡 请检查依赖是否正确安装") 