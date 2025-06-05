#!/usr/bin/env python3
"""
Chatterbox TTS 故障排除指南
常见问题的诊断和解决方案
"""

import torch
import sys
import subprocess
import pkg_resources

def check_system_requirements():
    """检查系统要求"""
    print("🔍 系统要求检查")
    print("="*50)
    
    # Python版本检查
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python版本过低，需要Python 3.8+")
    else:
        print("✅ Python版本符合要求")
    
    # PyTorch检查
    try:
        print(f"PyTorch版本: {torch.__version__}")
        print("✅ PyTorch已安装")
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    # 设备支持检查
    print("\n🖥️ 设备支持:")
    if torch.cuda.is_available():
        print(f"✅ CUDA可用: {torch.cuda.get_device_name()}")
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    elif torch.backends.mps.is_available():
        print("✅ Apple Metal (MPS) 可用")
    else:
        print("⚠️  仅CPU可用")
    
    return True

def check_dependencies():
    """检查依赖包"""
    print("\n📦 依赖包检查")
    print("="*50)
    
    required_packages = [
        "torch", "torchaudio", "transformers", 
        "diffusers", "omegaconf", "conformer",
        "safetensors", "librosa", "resampy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"✅ {package}: {version}")
        except pkg_resources.DistributionNotFound:
            print(f"❌ {package}: 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n🔧 安装缺失包命令:")
        print(f"pip install {' '.join(missing_packages)}")
    
    return len(missing_packages) == 0

def diagnose_common_issues():
    """诊断常见问题"""
    print("\n🔧 常见问题诊断")
    print("="*50)
    
    # 测试基本导入
    try:
        from chatterbox.tts import ChatterboxTTS
        print("✅ Chatterbox导入成功")
    except ImportError as e:
        print(f"❌ Chatterbox导入失败: {e}")
        print("💡 解决方案: pip install chatterbox-tts")
        return False
    
    # 测试模型加载
    try:
        device = "cpu"  # 使用CPU进行测试
        print("⏳ 测试模型加载...")
        model = ChatterboxTTS.from_pretrained(device=device)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("💡 可能的解决方案:")
        print("  • 检查网络连接（首次需要下载模型）")
        print("  • 清除缓存: rm -rf ~/.cache/huggingface/")
        print("  • 尝试手动下载模型")
        return False
    
    return True

def memory_optimization_guide():
    """内存优化指南"""
    print("\n💾 内存优化指南")
    print("="*50)
    
    print("🔧 减少内存使用:")
    print("• 使用混合精度: model.half() (仅GPU)")
    print("• 减小batch_size")
    print("• 清理GPU缓存: torch.cuda.empty_cache()")
    print("• 关闭不必要的应用程序")
    print()
    print("💡 推荐配置:")
    print("• 最小内存: 8GB RAM")
    print("• 推荐内存: 16GB+ RAM")
    print("• GPU内存: 6GB+ VRAM")

def performance_optimization():
    """性能优化建议"""
    print("\n🚀 性能优化建议")
    print("="*50)
    
    print("⚡ 加速推理:")
    print("• 使用GPU加速（CUDA/MPS）")
    print("• 启用混合精度训练")
    print("• 预加载模型避免重复加载")
    print("• 使用适当的temperature和cfg_weight")
    print()
    print("🎯 质量优化:")
    print("• 使用高质量参考音频")
    print("• 调整exaggeration参数")
    print("• 选择合适的采样率")
    print("• 后处理音频降噪")

def network_troubleshooting():
    """网络问题排除"""
    print("\n🌐 网络问题排除")
    print("="*50)
    
    print("📡 模型下载问题:")
    print("• 检查网络连接")
    print("• 设置代理: export https_proxy=http://proxy:port")
    print("• 使用国内镜像源")
    print("• 手动下载模型文件")
    print()
    print("🔄 替代下载方式:")
    print("• 使用HuggingFace镜像")
    print("• 离线模式运行")
    print("• 从本地路径加载模型")

def create_conda_env_script():
    """创建conda环境脚本"""
    script_content = """#!/bin/bash
# Chatterbox TTS Conda环境设置脚本

echo "🐍 创建Chatterbox TTS环境..."

# 创建conda环境
conda create -n chatterbox python=3.10 -y

# 激活环境
conda activate chatterbox

# 安装PyTorch (根据系统选择)
echo "🔧 安装PyTorch..."
# CUDA版本
# conda install pytorch torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# CPU版本
# conda install pytorch torchaudio cpuonly -c pytorch -y

# Mac版本 (MPS支持)
# conda install pytorch torchaudio -c pytorch -y

# 安装Chatterbox TTS
echo "📦 安装Chatterbox TTS..."
pip install chatterbox-tts

echo "✅ 安装完成！"
echo "💡 使用方法: conda activate chatterbox"
"""
    
    with open("setup_chatterbox_env.sh", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("📝 已创建环境设置脚本: setup_chatterbox_env.sh")
    print("🚀 运行命令: bash setup_chatterbox_env.sh")

def run_full_diagnosis():
    """运行完整诊断"""
    print("🏥 Chatterbox TTS 完整诊断")
    print("="*60)
    
    success = True
    success &= check_system_requirements()
    success &= check_dependencies()
    success &= diagnose_common_issues()
    
    print("\n" + "="*60)
    if success:
        print("🎉 所有检查通过！系统配置正常")
    else:
        print("⚠️  发现问题，请参考上述建议进行修复")
    
    print("\n📚 更多帮助:")
    memory_optimization_guide()
    performance_optimization()
    network_troubleshooting()
    create_conda_env_script()

if __name__ == "__main__":
    run_full_diagnosis() 