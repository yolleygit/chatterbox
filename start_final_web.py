#!/usr/bin/env python3
"""
最终版语音克隆网页界面启动脚本
解决进度同步和音频库管理问题
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🎭 最终版语音克隆网页界面")
    print("=" * 60)
    print("🎯 核心问题解决:")
    print("  • ✅ 进度条与AI生成过程实时同步")
    print("  • ✅ 详细显示每个处理阶段")
    print("  • ✅ 录音自动保存到音频库")
    print("  • ✅ 历史录音管理和重复使用")
    print()
    print("🆕 主要功能:")
    print("  • 🎙️ 智能音频库：自动保存、管理、重复使用")
    print("  • ⏳ 真实进度：前台进度与后台AI同步")
    print("  • 📊 详细统计：显示实际生成耗时")
    print("  • 🔄 完整流程：录音→保存→选择→克隆")
    print("  • 🎛️ 参数预设：一键设置最佳参数")
    print()
    print("🌐 访问地址:")
    print("  • 本地访问: http://localhost:7862")
    print("  • 局域网访问: http://0.0.0.0:7862")
    print("  • (使用7862端口避免冲突)")
    print()
    print("💡 使用提示:")
    print("  • AI生成过程需要时间，后台sampling是正常现象")
    print("  • 进度条现在与实际AI计算同步")
    print("  • 录音可保存到音频库，方便重复使用")
    print("  • 建议录制3-10秒高质量音频获得最佳效果")
    print()
    
    try:
        from examples.gradio_demos.final_voice_cloning_web import main as final_main
        print("⏳ 正在启动最终版服务器...")
        print("🔄 如需停止，请按 Ctrl+C")
        print()
        
        final_main()
        
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