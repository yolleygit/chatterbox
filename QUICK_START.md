# 🚀 Chatterbox TTS 快速开始指南

## 📋 系统要求

- **Python**: 3.8+
- **系统**: macOS, Linux, Windows
- **内存**: 8GB+ 推荐
- **存储**: 5GB+ 可用空间

## ⚡ 3分钟快速体验

### 1️⃣ 环境准备
```bash
# 创建conda环境
conda create -n chatterbox python=3.10
conda activate chatterbox

# 安装依赖
pip install -r requirements_web.txt
```

### 2️⃣ 启动Web界面（推荐）
```bash
# 🎯 核心命令 - 启动最终版Web界面
python start_final_web.py

# 浏览器访问: http://localhost:7862
```

### 3️⃣ 开始使用
1. 点击"🔧 加载AI模型"
2. 录制或上传音频（3-10秒）
3. 输入要合成的文本
4. 点击"🎭 开始语音克隆"

🎉 **就这么简单！** 您已经成功体验了AI语音克隆！

---

## 📱 其他启动选项

如果您想尝试不同的界面版本：

```bash
# 简化版界面（端口7860）
python start_simple_web.py

# 增强版界面（端口7861）  
python start_enhanced_web.py

# 语音克隆专版（端口7863）
python start_voice_cloning_web.py
```

## 🍎 Mac用户专用

Mac用户如遇到设备相关问题，请使用：
```bash
# MPS设备检测
python test_mps_device.py

# 查看Mac专用指南
open docs/Mac完整使用指南.md
```

## 🔧 遇到问题？

### 快速诊断
```bash
# 检查环境是否正确
conda list | grep torch

# 测试基础功能
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### 常见解决方案

| 问题 | 解决方法 |
|------|----------|
| **端口被占用** | 更改端口或关闭占用程序 |
| **模型下载失败** | 检查网络连接，重试 |
| **CUDA/MPS错误** | 使用CPU模式或重启 |
| **依赖缺失** | 重新安装：`pip install -r requirements_web.txt` |

## 📚 深入学习

想了解更多功能？查看完整文档：

- 📖 **[完整文档](docs/README.md)** - 全面的使用指南
- 🍎 **[Mac指南](docs/Mac完整使用指南.md)** - Mac系统专用
- 🎭 **[语音克隆](docs/语音克隆完整指南.md)** - 详细教程
- 🔧 **[问题解决](docs/常见问题解决方案.md)** - 疑难排除

## 💡 使用技巧

- 🎯 **推荐使用最终版**：`python start_final_web.py`
- 🎤 **录音质量**：3-10秒清晰语音效果最佳
- 🌐 **网络环境**：首次使用需下载模型（约2GB）
- 💾 **音频管理**：支持保存和重复使用录音

---

🎭 **立即开始您的AI语音克隆之旅！**

**核心命令**: `conda activate chatterbox && python start_final_web.py` 