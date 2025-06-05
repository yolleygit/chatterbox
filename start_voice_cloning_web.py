#!/usr/bin/env python3
"""
语音克隆网页界面启动脚本
"""

import sys
import os
import webbrowser
import time

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🎭 语音克隆网页界面启动器")
    print("=" * 50)
    print("📱 功能特性:")
    print("  • 🎙️ 麦克风录音/文件上传")
    print("  • ⚙️ 参数实时调整")
    print("  • 🎭 AI语音克隆")
    print("  • 📊 结果分析显示")
    print("  • 💾 音频文件下载")
    print()
    
    print("🌐 访问地址:")
    print("  • 本地访问: http://localhost:7860")
    print("  • 局域网访问: http://0.0.0.0:7860")
    print()
    
    print("📖 使用步骤:")
    print("  1. 上传3-10秒清晰音频文件")
    print("  2. 输入想要合成的文本内容")
    print("  3. 调整参数(相似度/表达度/随机性)")
    print("  4. 点击'加载模型'(首次使用)")
    print("  5. 点击'开始语音克隆'生成结果")
    print()
    
    try:
        from examples.gradio_demos.simple_voice_cloning_web import main as web_main
        print("⏳ 正在启动网页服务器...")
        print("💡 启动完成后会自动打开浏览器")
        print("🔄 如需停止，请按 Ctrl+C")
        print()
        
        web_main()
        
    except KeyboardInterrupt:
        print("\n👋 用户中断，关闭服务器")
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("💡 请确保安装了gradio: pip install gradio")
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("💡 请检查环境配置和依赖安装")

if __name__ == "__main__":
    main() 