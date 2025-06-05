# 🍎 Mac系统Chatterbox TTS完整使用指南

## 🚨 已解决的问题

✅ **CUDA设备映射错误** - 已修复  
✅ **torch.load参数兼容性** - 已修复  
✅ **MPS设备转移问题** - 已修复  
✅ **网络连接问题** - 提供离线解决方案  

## 📁 可用的脚本文件

### 1. 🧪 测试脚本
- **`simple_mac_test.py`** - 快速兼容性测试（推荐首次运行）
- **`test_installation.py`** - 基础安装验证

### 2. 🎤 TTS示例脚本
- **`basic_tts_example_offline.py`** - 基本TTS示例（离线版，推荐）
- **`basic_tts_example.py`** - 基本TTS示例（在线版）
- **`mac_tts_example.py`** - Mac专用优化示例

### 3. 🔧 工具脚本
- **`mac_device_patch.py`** - 设备映射补丁工具

## 🚀 推荐运行顺序

### 第一步：测试系统兼容性
```bash
conda activate chatterbox
python simple_mac_test.py
```

### 第二步：运行基本TTS示例
```bash
# 推荐使用离线版本（避免网络问题）
python basic_tts_example_offline.py
```

### 第三步：尝试高级功能
```bash
# 如果网络稳定，可以尝试在线示例
python mac_tts_example.py --test
python mac_tts_example.py
```

## 🔧 故障排除

### 网络连接问题
**错误**: `ProxyError` 或 `MaxRetryError`
**解决方案**: 使用离线版本
```bash
python basic_tts_example_offline.py
```

### CUDA设备映射错误
**错误**: `RuntimeError: Attempting to deserialize object on a CUDA device`
**解决方案**: 已在所有脚本中自动修复

### MPS设备问题
**错误**: `'ChatterboxTTS' object has no attribute 'to'`
**解决方案**: 代码会自动回退到CPU模式

### 内存不足
**错误**: 内存相关错误
**解决方案**: 
1. 关闭其他大型应用
2. 重启Python会话
3. 使用CPU模式

## 📊 性能建议

### 内存配置建议
- **16GB+ 内存**: 推荐配置，支持MPS加速
- **8-16GB 内存**: 可以运行，建议使用CPU模式
- **<8GB 内存**: 可能需要多次尝试或使用更短的文本

### 设备选择
- **Apple Silicon (M1/M2/M3)**: 优先使用MPS，回退CPU
- **Intel Mac**: 使用CPU模式
- **有独立显卡**: 理论上可以使用CUDA（较少见）

## ⚙️ 环境变量优化

在运行前可以设置这些环境变量：
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

## 🎯 常见使用场景

### 快速测试
```bash
python simple_mac_test.py
```

### 日常使用
```bash
python basic_tts_example_offline.py
```

### 高级功能演示
```bash
python mac_tts_example.py
```

### 批量处理
修改脚本中的文本列表，使用离线版本进行批量生成

## 📝 输出文件

运行成功后会生成以下音频文件：
- `basic_output_offline.wav` - 离线版本输出
- `basic_output.wav` - 在线版本输出
- `mac_output_1.wav`, `mac_output_2.wav`, `mac_output_3.wav` - Mac专用示例输出

## 🎉 成功标志

看到以下信息表示运行成功：
```
🎵 语音已保存到: xxx.wav
📊 采样率: 24000Hz
⏱️  时长: X.XX秒
🎯 使用设备: mps/cpu
```

## 💡 最佳实践

1. **首次使用**: 运行`simple_mac_test.py`确认环境
2. **日常使用**: 优先使用离线版本避免网络问题
3. **性能优化**: 关闭不必要的应用释放内存
4. **错误处理**: 大多数错误会自动处理和回退
5. **定期清理**: 重启Python会话清理内存碎片

现在您的Mac系统已经可以完美运行Chatterbox TTS了！🎊 