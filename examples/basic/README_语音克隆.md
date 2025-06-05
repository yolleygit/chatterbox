# 语音克隆使用说明

## 🎭 语音克隆简介

语音克隆是通过分析参考音频文件，学习其声音特征，然后用这个声音来合成新文本的技术。

## 🚀 快速开始

### 方法1: 使用现有音频文件
```bash
# 运行现成的演示（使用项目中已有的音频文件）
python examples/basic/voice_cloning_with_existing_audio.py
```

### 方法2: 使用自己的音频文件
```bash
# 运行快速克隆（需要自己提供音频文件）
python examples/basic/quick_voice_cloning.py
```

## 📝 核心代码

```python
from chatterbox.tts import ChatterboxTTS

# 加载模型
model = ChatterboxTTS.from_pretrained(device="cpu")

# 语音克隆（核心代码）
wav = model.generate(
    text="要合成的文本",
    audio_prompt_path="参考音频.wav",  # 关键：指定参考音频
    cfg_weight=0.8,       # 相似度控制 (0.1-1.0)
    exaggeration=0.7,     # 情感表达 (0.1-2.0) 
    temperature=0.7       # 随机性控制 (0.1-2.0)
)

# 保存结果
import torchaudio as ta
ta.save("克隆结果.wav", wav, model.sr)
```

## 🎛️ 关键参数

| 参数 | 作用 | 推荐值 | 效果 |
|------|------|--------|------|
| `audio_prompt_path` | 参考音频文件路径 | 必填 | 决定克隆的目标声音 |
| `cfg_weight` | 相似度控制 | 0.7-0.8 | 越高越像参考音频 |
| `exaggeration` | 情感表达度 | 0.5-0.8 | 控制语音的表现力 |
| `temperature` | 随机性控制 | 0.6-1.0 | 控制生成的多样性 |

## 📁 音频文件要求

### ✅ 推荐规格
- **格式**: WAV, MP3, FLAC, M4A
- **时长**: 3-10秒（最佳5-8秒）
- **质量**: 清晰、无背景噪音
- **采样率**: 16kHz或更高
- **内容**: 自然说话，避免朗读腔

### 🎙️ 录制建议
1. 使用Mac的QuickTime Player或语音备忘录
2. 安静环境录制
3. 正常语速和音量
4. 录制完整句子
5. 保存为WAV格式

## 🎯 参数调优指南

### 高质量克隆（最像原声）
```python
cfg_weight=0.9,      # 高相似度
exaggeration=0.5,    # 低表达度
temperature=0.6      # 低随机性
```

### 表演创作（表现力强）
```python
cfg_weight=0.6,      # 中等相似度
exaggeration=1.2,    # 高表达度
temperature=1.0      # 中等随机性
```

### 稳定输出（一致性好）
```python
cfg_weight=0.8,      # 高相似度
exaggeration=0.4,    # 低表达度
temperature=0.5      # 低随机性
```

## 📊 演示结果

✅ **成功案例**: 使用 `audio_output/basic_output_offline.wav` 作为参考，成功生成了4个语音克隆文件：
- `cloned_from_existing_1.wav` (3.04秒)
- `cloned_from_existing_2.wav` (3.64秒)
- `cloned_from_existing_3.wav` (3.52秒)
- `cloned_from_existing_4.wav` (2.76秒)

## 🔧 常见问题解决

### Q: 生成的声音不像参考音频？
A: 提高 `cfg_weight` 到 0.8-0.9，降低 `temperature` 到 0.5-0.7

### Q: 声音听起来很机械？
A: 提高 `exaggeration` 到 0.6-0.8，使用更自然的参考音频

### Q: 音频文件格式不支持？
A: 使用以下代码转换：
```python
import librosa
import soundfile as sf
audio, sr = librosa.load("input.mp3", sr=22050)
sf.write("output.wav", audio, sr)
```

## 🌟 应用场景

- **📖 有声书制作**: 用自己的声音朗读书籍
- **🎬 视频配音**: 为视频角色配音
- **🤖 个人语音助手**: 创建个性化AI助手
- **🎵 创意内容**: 音乐、播客制作
- **🌍 多语言配音**: 本地化内容制作

## 📚 相关资源

- **详细教程**: `examples/advanced/voice_cloning_tutorial.py`
- **完整指南**: `docs/语音克隆完整指南.md`
- **中文语音**: `examples/basic/chinese_tts_example.py`
- **参数对比**: 运行不同参数组合，听效果差异

## ⚖️ 使用须知

⚠️ **重要提醒**:
- 仅用于合法和道德目的
- 使用他人声音需获得同意
- 不得用于欺诈或误导
- 遵守当地法律法规

## 🎉 开始体验

现在就试试语音克隆功能吧！从运行演示开始：

```bash
# 使用现有音频演示
python examples/basic/voice_cloning_with_existing_audio.py

# 或使用自己的音频
python examples/basic/quick_voice_cloning.py
```

祝您语音克隆体验愉快！🎊 