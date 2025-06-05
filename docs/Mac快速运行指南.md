# 🍎 Mac系统Chatterbox TTS快速运行指南

## 问题解决方案

您遇到的`RuntimeError: Attempting to deserialize object on a CUDA device`错误已修复！

## 修复后的运行步骤

### 1. 使用修复后的基本示例
```bash
python basic_tts_example.py
```

### 2. 使用Mac专用优化示例（推荐）
```bash
python mac_tts_example.py
```

### 3. 快速兼容性测试
```bash
python mac_tts_example.py --test
```

### 4. 应用系统补丁（独立运行）
```bash
python mac_device_patch.py
```

## 已解决的问题

✅ **CUDA设备映射错误** - 自动映射到CPU/MPS
✅ **MPS设备兼容性** - Apple Silicon优化
✅ **内存管理优化** - 智能缓存清理
✅ **自动设备回退** - 错误时自动降级

## Mac特定优化

🚀 **Apple Silicon加速** - M1/M2/M3芯片硬件加速
💾 **智能内存管理** - 自动清理MPS缓存
🔄 **自动错误恢复** - 遇到问题时优雅降级
📊 **性能监控** - 实时因子和耗时统计

## 推荐使用流程

1. **首次运行**：使用`mac_tts_example.py`进行完整测试
2. **日常使用**：使用修复后的`basic_tts_example.py`
3. **问题排查**：运行快速测试模式

## 性能建议

- **16GB+ 内存**：推荐配置，支持MPS加速
- **8GB 内存**：建议使用CPU模式
- **关闭大型应用**：释放内存供AI推理使用
- **定期重启Python**：清理内存碎片

现在可以安全地在您的Mac上运行Chatterbox TTS了！🎉 