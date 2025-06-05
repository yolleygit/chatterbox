# 🍎 Mac MPS设备配置指南

## 📋 问题解决总结

### ❌ **原始问题**
`start_final_web.py` 启动的Web界面一直使用CPU而不是MPS设备。

### ✅ **解决方案**
修改了 `examples/gradio_demos/final_voice_cloning_web.py` 中的设备选择逻辑。

## 🔧 具体修改内容

### 1. **设备智能选择逻辑**

**原始代码（第288行）：**
```python
# 加载模型
model = ChatterboxTTS.from_pretrained(device="cpu")

# 获取模型信息  
device = "CPU (Apple Silicon MPS)" if torch.backends.mps.is_available() else "CPU"
```

**修改后代码：**
```python
# 智能设备选择
if torch.backends.mps.is_available():
    device_name = "mps"
    device_display = "MPS (Apple Silicon)"
elif torch.cuda.is_available():
    device_name = "cuda"
    device_display = "CUDA (NVIDIA GPU)"
else:
    device_name = "cpu"
    device_display = "CPU"

# 加载模型到最佳设备
model = ChatterboxTTS.from_pretrained(device=device_name)

# 获取模型信息
device = device_display
```

### 2. **启动脚本增强**

在 `start_final_web.py` 中添加了设备检测提示：
```python
# 显示设备检测信息
try:
    import torch
    if torch.backends.mps.is_available():
        print("  • 🚀 检测到Apple Silicon MPS，将使用GPU加速")
    elif torch.cuda.is_available():
        print("  • 🚀 检测到CUDA GPU，将使用GPU加速")
    else:
        print("  • 💻 将使用CPU运行（性能可能较慢）")
except:
    pass
```

## 🧪 验证测试

### **MPS兼容性测试结果**
```bash
python test_mps_device.py
```

**测试结果：**
- ✅ **MPS基础功能**: 可用
- ✅ **Chatterbox MPS**: 兼容  
- 🎉 **推荐设置**: `device='mps'` - 使用Apple Silicon GPU加速

### **关键测试输出**
```
📋 PyTorch版本: 2.6.0
🍎 MPS可用: True
🏗️ MPS已构建: True
✅ MPS设备可用！
✅ Chatterbox TTS在MPS上加载成功！
✅ 生成成功！音频长度: 1.84秒
```

## 🚀 性能对比

| 设备类型 | 加载速度 | 生成速度 | 内存使用 | 推荐度 |
|---------|---------|---------|---------|--------|
| **MPS (Apple Silicon)** | 快 | 快 | 中等 | ⭐⭐⭐⭐⭐ |
| CPU | 中等 | 慢 | 低 | ⭐⭐⭐ |
| CUDA (如果有) | 快 | 最快 | 高 | ⭐⭐⭐⭐⭐ |

## 📊 现在的工作流程

1. **启动检测**：系统自动检测可用设备
2. **智能选择**：优先级 MPS > CUDA > CPU
3. **模型加载**：自动加载到最优设备
4. **Web界面**：显示当前使用的设备信息

## 🎯 验证方法

### **方法1：查看启动信息**
启动时会显示：
```
• 🚀 检测到Apple Silicon MPS，将使用GPU加速
```

### **方法2：Web界面模型信息**
在Web界面点击"🔧 加载模型"后会显示：
```
📋 模型详细信息:
• 💻 运行设备: MPS (Apple Silicon)
```

### **方法3：运行测试脚本**
```bash
python test_mps_device.py
```

## 💡 使用建议

### **✅ 推荐配置**
- **Mac用户**: 使用MPS设备，获得最佳性能
- **启动命令**: `python start_final_web.py`
- **访问地址**: `http://localhost:7862`

### **🎛️ 参数建议**
在MPS设备上，建议使用以下参数获得最佳效果：
- **相似度控制**: 0.7-0.8
- **情感表达**: 0.6-0.8  
- **随机性控制**: 0.7-0.9

## 🔧 故障排除

### **如果MPS仍未生效**
1. **检查环境**：确保使用 `conda activate chatterbox`
2. **重新安装**：重新安装PyTorch MPS版本
3. **手动测试**：运行 `python test_mps_device.py`
4. **查看日志**：检查Web界面的模型加载信息

### **性能优化**
```python
# 如果需要手动设置
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
```

## 📈 预期改进

使用MPS设备后，您应该观察到：
- 🚀 **模型加载更快**
- ⚡ **语音生成速度提升**  
- 📊 **实时因子改善**
- 💻 **CPU占用率降低**

---

**✅ 问题已完全解决！** 现在您的最终版Web界面会自动使用Apple Silicon MPS加速，获得最佳的语音克隆性能！ 