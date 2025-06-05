#!/usr/bin/env python3
"""
增强版语音克隆网页界面启动脚本
解决所有用户反馈的问题
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🎭 增强版语音克隆网页界面")
    print("=" * 50)
    print("🆕 新增功能:")
    print("  • 🎙️ 音频预览和分析")
    print("  • 🔧 详细模型信息显示")
    print("  • ⏳ 实时进度反馈")
    print("  • 🛠️ 智能错误处理")
    print("  • 🎛️ 参数预设快捷按钮")
    print()
    
    print("🌐 访问地址:")
    print("  • 本地访问: http://localhost:7861")
    print("  • 局域网访问: http://0.0.0.0:7861")
    print("  • (使用7861端口避免与简单版冲突)")
    print()
    
    print("🔧 问题修复:")
    print("  • ✅ 修复'index out of range'错误")
    print("  • ✅ 增加音频预览播放功能")
    print("  • ✅ 显示详细模型加载信息")
    print("  • ✅ 添加语音克隆进度条")
    print()
    
    try:
        from examples.gradio_demos.enhanced_voice_cloning_web import main as enhanced_main
        print("⏳ 正在启动增强版服务器...")
        print("🔄 如需停止，请按 Ctrl+C")
        print()
        
        enhanced_main()
        
    except KeyboardInterrupt:
        print("\n👋 用户中断，关闭服务器")
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("💡 请确保安装了所有依赖:")
        print("   pip install gradio librosa")
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("💡 请检查环境配置")

if __name__ == "__main__":
    main() 