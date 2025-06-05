# 🔰 Basic Examples - 基础示例

本目录包含Chatterbox TTS的基础使用示例，适合初学者快速上手。

## 📄 文件说明

### 🎤 `basic_tts_example.py`
**最简单的TTS使用示例**
- 功能: 文本转语音的基本演示
- 特点: 代码简洁，容易理解
- 适用: 所有平台（包含Mac优化）
- 输出: `basic_output.wav`

```bash
python basic_tts_example.py
```

### 💾 `basic_tts_example_offline.py` ⭐ **推荐**
**离线版TTS示例**
- 功能: 使用本地缓存，避免网络问题
- 特点: 更稳定，适合网络不佳的环境
- 优势: 自动错误恢复，内存优化
- 输出: `basic_output_offline.wav`

```bash
python basic_tts_example_offline.py
```

### 🎵 `example_tts.py`
**官方基础TTS示例**
- 功能: 原始的官方示例代码
- 特点: 简洁的英文代码
- 用途: 参考官方实现方式

```bash
python example_tts.py
```

### 🔄 `example_vc.py`
**基础语音转换示例**
- 功能: 将一个语音转换为另一个人的声音
- 需要: 输入音频文件 + 目标声音文件
- 输出: 转换后的语音

```bash
python example_vc.py
```

## 🚀 快速开始

### 第一次使用
1. 首先运行离线版本（推荐）:
   ```bash
   python basic_tts_example_offline.py
   ```

2. 如果成功，尝试在线版本:
   ```bash
   python basic_tts_example.py
   ```

### Mac用户特别提示
- 这些示例已包含Mac兼容性优化
- 如果遇到问题，请使用 `../mac_specific/` 目录中的专用示例

## 📋 参数说明

### 基本参数
- **text**: 要合成的文本内容
- **device**: 计算设备 ("cpu", "mps", "cuda")
- **audio_prompt_path**: 参考音频路径（可选）

### 生成参数
- **exaggeration**: 情感夸张度 (0.1-2.0)
- **cfg_weight**: 一致性权重 (0.1-1.0) 
- **temperature**: 随机性控制 (0.1-2.0)

## 💡 使用技巧

1. **首次运行**: 需要下载模型，请耐心等待
2. **内存不足**: 关闭其他应用，或重启Python会话
3. **网络问题**: 优先使用离线版本
4. **设备选择**: 系统会自动选择最佳设备

## 🎯 预期输出

运行成功后，你将看到:
```
🎤 初始化Chatterbox TTS模型...
🍎 使用Apple Metal加速  # (Mac用户)
✅ 模型加载完成
📝 合成文本: 你好，我是Chatterbox...
⏳ 正在生成语音...
🎵 语音已保存到: basic_output.wav
📊 采样率: 24000Hz
⏱️  时长: 3.42秒
```

## ❓ 常见问题

**Q: 提示CUDA错误怎么办？**
A: 这是正常的，系统会自动回退到CPU模式

**Q: 生成的音频时长很短怎么办？**  
A: 尝试使用更长的文本，或调整参数

**Q: Mac上运行很慢怎么办？**
A: 使用 `../mac_specific/mac_tts_example.py` 获得更好性能 