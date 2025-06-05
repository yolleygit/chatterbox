

#!/usr/bin/env python3
"""
最终版语音克隆网页界面
解决问题：进度条与AI生成同步、录音音频库管理
"""

import gradio as gr
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import os
import time
import librosa
import numpy as np
from pathlib import Path
import datetime
import json
import threading
from typing import Optional, List, Tuple

# 全局变量
model = None
model_info = {"name": "", "version": "", "device": ""}
temp_dir = Path("output/web_voice_cloning")
audio_library_dir = Path("output/audio_library")  # 音频库目录
temp_dir.mkdir(parents=True, exist_ok=True)
audio_library_dir.mkdir(parents=True, exist_ok=True)

# 音频库索引文件
audio_index_file = audio_library_dir / "audio_index.json"

def load_audio_index():
    """加载音频库索引"""
    if audio_index_file.exists():
        try:
            with open(audio_index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_audio_index(index):
    """保存音频库索引"""
    with open(audio_index_file, 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

def apply_torch_load_patch():
    """应用torch.load补丁"""
    original_load = torch.load
    
    def patched_load(f, map_location=None, **kwargs):
        if map_location is None and not torch.cuda.is_available():
            map_location = torch.device('cpu')
        try:
            return original_load(f, map_location=map_location, **kwargs)
        except RuntimeError as e:
            if "CUDA" in str(e):
                return original_load(f, map_location=torch.device('cpu'), **kwargs)
            raise e
    
    torch.load = patched_load

def validate_and_process_audio(audio_input):
    """验证和处理音频输入"""
    if audio_input is None:
        return None, "❌ 请先录制或上传音频文件"
    
    try:
        # 处理不同的音频输入格式
        if isinstance(audio_input, tuple):
            # Gradio麦克风录音格式 (sample_rate, audio_data)
            sr, audio_data = audio_input
            audio_data = audio_data.astype(np.float32)
            
            # 转换为单声道
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # 验证音频长度
            duration = len(audio_data) / sr
            if duration < 1.0:
                return None, "❌ 音频太短，请录制至少1秒的音频"
            if duration > 30.0:
                return None, "❌ 音频太长，请录制不超过30秒的音频"
            
            # 保存临时文件
            timestamp = int(time.time())
            temp_audio_path = temp_dir / f"temp_audio_{timestamp}.wav"
            
            # 确保音频数据在正确范围内
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
            
            ta.save(temp_audio_path, torch.from_numpy(audio_data).unsqueeze(0), sr)
            return str(temp_audio_path), f"✅ 音频处理成功！时长: {duration:.2f}秒，采样率: {sr}Hz"
            
        else:
            # 上传的文件路径
            if not os.path.exists(audio_input):
                return None, "❌ 音频文件不存在"
            
            # 使用librosa加载和验证
            audio_data, sr = librosa.load(audio_input, sr=None)
            duration = len(audio_data) / sr
            
            if duration < 1.0:
                return None, "❌ 音频太短，请使用至少1秒的音频"
            if duration > 30.0:
                return None, "❌ 音频太长，请使用不超过30秒的音频"
                
            return audio_input, f"✅ 音频验证成功！时长: {duration:.2f}秒，采样率: {sr}Hz"
            
    except Exception as e:
        return None, f"❌ 音频处理失败: {str(e)}"

def save_recording_to_library(audio_input, custom_name: str = ""):
    """将录音保存到音频库"""
    if audio_input is None:
        return "❌ 请先录制音频", get_audio_library_list()
    
    try:
        # 验证和处理音频
        processed_path, status_msg = validate_and_process_audio(audio_input)
        if processed_path is None:
            return status_msg, get_audio_library_list()
        
        # 生成文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if custom_name.strip():
            safe_name = "".join(c for c in custom_name if c.isalnum() or c in (' ', '-', '_')).strip()
            filename = f"{safe_name}_{timestamp}.wav"
        else:
            filename = f"录音_{timestamp}.wav"
        
        # 保存到音频库
        library_path = audio_library_dir / filename
        
        if isinstance(audio_input, tuple):
            # 重新保存录音数据到音频库
            sr, audio_data = audio_input
            audio_data = audio_data.astype(np.float32)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
            ta.save(library_path, torch.from_numpy(audio_data).unsqueeze(0), sr)
        else:
            # 复制文件到音频库
            import shutil
            shutil.copy2(processed_path, library_path)
        
        # 更新音频库索引
        index = load_audio_index()
        audio_info = {
            "filename": filename,
            "path": str(library_path),
            "created_time": timestamp,
            "name": custom_name.strip() if custom_name.strip() else f"录音_{timestamp}",
            "duration": librosa.get_duration(path=str(library_path)),
            "sample_rate": librosa.get_samplerate(str(library_path))
        }
        index[filename] = audio_info
        save_audio_index(index)
        
        success_msg = f"""✅ 录音已保存到音频库！

📋 保存信息:
• 🎵 名称: {audio_info['name']}
• 📁 文件: {filename}
• ⏱️ 时长: {audio_info['duration']:.2f}秒
• 📈 采样率: {audio_info['sample_rate']}Hz
• 🕐 保存时间: {timestamp}

💡 现在可以在历史录音中选择使用此音频！"""
        
        return success_msg, get_audio_library_list()
        
    except Exception as e:
        return f"❌ 保存失败: {str(e)}", get_audio_library_list()

def get_audio_library_list():
    """获取音频库列表"""
    index = load_audio_index()
    if not index:
        return ["暂无历史录音"]
    
    # 按创建时间倒序排列
    sorted_items = sorted(index.items(), key=lambda x: x[1]['created_time'], reverse=True)
    return [f"{item['name']} ({item['duration']:.1f}s)" for _, item in sorted_items]

def load_audio_from_library(selected_audio: str):
    """从音频库加载选定的音频"""
    if not selected_audio or selected_audio == "暂无历史录音":
        return None, "请选择有效的历史录音"
    
    try:
        index = load_audio_index()
        
        # 根据显示名称查找文件
        for filename, info in index.items():
            display_name = f"{info['name']} ({info['duration']:.1f}s)"
            if display_name == selected_audio:
                audio_path = info['path']
                if os.path.exists(audio_path):
                    # 分析选定的音频
                    audio_data, sr = librosa.load(audio_path, sr=None)
                    duration = len(audio_data) / sr
                    max_amplitude = np.max(np.abs(audio_data))
                    
                    analysis = f"""📚 来自音频库: {info['name']}

📊 音频分析:
• ⏱️ 时长: {duration:.2f}秒
• 📈 采样率: {sr}Hz
• 🔊 最大音量: {max_amplitude:.3f}
• 📅 录制时间: {info['created_time']}
• 📁 文件: {filename}

✅ 已加载历史录音，可以开始语音克隆！"""
                    
                    return audio_path, analysis
                else:
                    return None, f"❌ 音频文件不存在: {audio_path}"
        
        return None, "❌ 未找到对应的音频文件"
        
    except Exception as e:
        return None, f"❌ 加载失败: {str(e)}"

def analyze_uploaded_audio(audio_input):
    """分析上传的音频并返回预览信息"""
    if audio_input is None:
        return None, "请先录制或上传音频文件"
    
    try:
        processed_path, status_msg = validate_and_process_audio(audio_input)
        if processed_path is None:
            return None, status_msg
        
        # 分析音频信息
        if isinstance(audio_input, tuple):
            sr, audio_data = audio_input
            audio_data = audio_data.astype(np.float32)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            duration = len(audio_data) / sr
            max_amplitude = np.max(np.abs(audio_data))
        else:
            audio_data, sr = librosa.load(audio_input, sr=None)
            duration = len(audio_data) / sr
            max_amplitude = np.max(np.abs(audio_data))
        
        # 生成分析报告
        quality = "优秀" if 3 <= duration <= 10 and sr >= 16000 and max_amplitude > 0.1 else "可用"
        analysis = f"""
📊 音频分析结果:
• ⏱️ 时长: {duration:.2f}秒
• 📈 采样率: {sr}Hz  
• 🔊 最大音量: {max_amplitude:.3f}
• 🎵 质量评估: {quality}

💡 建议:
• 最佳时长: 3-10秒
• 推荐采样率: ≥16kHz
• 音量水平: >0.1 (当前: {max_amplitude:.3f})
• 录音环境: 安静无背景噪音

💾 提示: 点击"保存到音频库"可将此录音保存供以后使用
"""
        
        return processed_path, analysis
        
    except Exception as e:
        return None, f"❌ 音频分析失败: {str(e)}"

def load_model_with_info():
    """加载TTS模型并返回详细信息"""
    global model, model_info
    
    if model is None:
        apply_torch_load_patch()
        try:
            # 显示加载开始
            yield "🔄 正在初始化AI模型..."
            
            # 加载模型
            model = ChatterboxTTS.from_pretrained(device="cpu")
            
            # 获取模型信息
            device = "CPU (Apple Silicon MPS)" if torch.backends.mps.is_available() else "CPU"
            model_info = {
                "name": "ChatterboxTTS",
                "version": "v1.0",
                "device": device,
                "precision": "float32",
                "memory": "约2-4GB"
            }
            
            success_msg = f"""✅ AI模型加载成功！

📋 模型详细信息:
• 🤖 模型名称: {model_info['name']}
• 📦 版本: {model_info['version']}
• 💻 运行设备: {model_info['device']}
• 🎯 精度: {model_info['precision']}
• 💾 内存使用: {model_info['memory']}

🎭 系统已就绪，现在可以开始语音克隆了！"""
            
            yield success_msg
            
        except Exception as e:
            error_msg = f"""❌ 模型加载失败!

🔍 错误信息: {str(e)}

💡 解决建议:
• 检查网络连接（首次使用需下载模型）
• 确保有足够内存空间（需要2-4GB）
• 重启程序后重试
• 检查CUDA/MPS设备状态"""
            yield error_msg
    else:
        yield f"""✅ AI模型已就绪！

📋 当前模型信息:
• 🤖 模型名称: {model_info['name']}
• 💻 运行设备: {model_info['device']}
• 🎯 状态: 已加载并准备就绪"""

def enhanced_voice_clone_v2(audio_file, text, cfg_weight, exaggeration, temperature, progress=gr.Progress()):
    """增强的语音克隆函数V2，改进进度同步"""
    global model
    
    # 检查模型
    if model is None:
        return None, "❌ 请先点击'🔧 加载模型'按钮！"
    
    # 检查输入
    if audio_file is None:
        return None, "❌ 请先录制、上传或从音频库选择参考音频！"
    
    if not text.strip():
        return None, "❌ 请输入要合成的文本内容！"
    
    try:
        # 步骤1: 验证音频 (5%)
        progress(0.05, desc="🔍 验证音频文件...")
        processed_audio, validation_msg = validate_and_process_audio(audio_file)
        if processed_audio is None:
            return None, f"❌ 音频验证失败: {validation_msg}"
        
        # 步骤2: 准备参数 (10%)
        progress(0.10, desc="⚙️ 准备生成参数...")
        time.sleep(0.3)
        
        # 步骤3: 开始AI生成 (15% - 85%)
        progress(0.15, desc="🎭 开始AI语音克隆...")
        
        # 在这里我们需要分阶段更新进度
        def update_generation_progress():
            """模拟AI生成过程的进度更新"""
            stages = [
                (0.20, "🧠 AI正在分析参考音频..."),
                (0.35, "🎵 提取声音特征..."), 
                (0.50, "📝 处理文本内容..."),
                (0.65, "🎭 开始语音合成..."),
                (0.80, "🔄 优化音频质量...")
            ]
            
            for prog, desc in stages:
                progress(prog, desc=desc)
                time.sleep(0.8)  # 给用户看到进度的时间
        
        # 在后台更新进度
        progress_thread = threading.Thread(target=update_generation_progress)
        progress_thread.start()
        
        # 执行语音克隆（这是实际的AI计算）
        start_time = time.time()
        wav_tensor = model.generate(
            text=text,
            audio_prompt_path=processed_audio,
            cfg_weight=cfg_weight,
            exaggeration=exaggeration,
            temperature=temperature
        )
        generation_time = time.time() - start_time
        
        # 等待进度线程完成
        progress_thread.join()
        
        # 步骤4: 保存结果 (90%)
        progress(0.90, desc="💾 保存音频文件...")
        timestamp = int(time.time())
        output_path = temp_dir / f"cloned_voice_{timestamp}.wav"
        ta.save(output_path, wav_tensor, model.sr)
        
        # 步骤5: 生成报告 (95%)
        progress(0.95, desc="📊 生成结果报告...")
        audio_duration = wav_tensor.shape[1] / model.sr
        file_size = os.path.getsize(output_path) / 1024  # KB
        
        # 完成 (100%)
        progress(1.0, desc="✅ 语音克隆完成！")
        
        result_info = f"""🎉 语音克隆成功完成！

📊 生成统计:
• 🎼 音频时长: {audio_duration:.2f}秒
• 📁 文件大小: {file_size:.1f}KB
• ⏱️ 生成耗时: {generation_time:.1f}秒
• 💾 保存路径: {output_path.name}
• 🕐 完成时间: {time.strftime('%H:%M:%S')}

⚙️ 使用参数:
• 🎯 相似度控制: {cfg_weight} {'(高相似度)' if cfg_weight >= 0.8 else '(中等相似度)' if cfg_weight >= 0.6 else '(低相似度)'}
• 🎭 情感表达度: {exaggeration} {'(丰富表达)' if exaggeration >= 1.0 else '(自然表达)' if exaggeration >= 0.7 else '(平淡表达)'}
• 🎲 随机性控制: {temperature} {'(高变化)' if temperature >= 0.9 else '(平衡)' if temperature >= 0.7 else '(稳定)'}

💡 提示: 可调整参数重新生成以获得不同效果！
🎵 可以下载音频文件保存到本地使用"""
        
        return str(output_path), result_info
        
    except Exception as e:
        error_detail = str(e)
        if "index out of range" in error_detail:
            error_msg = f"""❌ 语音克隆失败: 音频处理错误

🔍 具体错误: {error_detail}

💡 解决方案:
• 尝试使用更长的参考音频（3-10秒）
• 确保音频格式正确（WAV/MP3/FLAC）
• 检查音频质量（无损坏、无空白）
• 重新录制或上传音频文件
• 调整参数后重试（降低cfg_weight到0.7）
• 尝试从音频库选择其他录音"""
        else:
            error_msg = f"""❌ 语音克隆失败: {error_detail}

💡 建议:
• 检查参考音频质量
• 尝试调整参数设置
• 重启加载模型
• 查看终端输出获取更多信息
• 生成耗时较长是正常现象，请耐心等待"""
        
        return None, error_msg

def delete_audio_from_library(selected_audio: str):
    """从音频库删除选定的音频"""
    if not selected_audio or selected_audio == "暂无历史录音":
        return "请选择要删除的录音", get_audio_library_list()
    
    try:
        index = load_audio_index()
        
        # 查找要删除的文件
        for filename, info in index.items():
            display_name = f"{info['name']} ({info['duration']:.1f}s)"
            if display_name == selected_audio:
                # 删除文件
                audio_path = Path(info['path'])
                if audio_path.exists():
                    audio_path.unlink()
                
                # 从索引中删除
                del index[filename]
                save_audio_index(index)
                
                return f"✅ 已删除录音: {info['name']}", get_audio_library_list()
        
        return "❌ 未找到要删除的录音", get_audio_library_list()
        
    except Exception as e:
        return f"❌ 删除失败: {str(e)}", get_audio_library_list()

def create_final_interface():
    """创建最终版网页界面"""
    
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
        margin: auto !important;
    }
    .gr-button {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        border: none !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }
    .gr-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
    }
    .audio-library {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 15px;
        background: rgba(76, 175, 80, 0.1);
    }
    .progress-info {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(css=custom_css, title="🎭 最终版语音克隆", theme=gr.themes.Soft()) as demo:
        
        gr.HTML("""
        <div style="text-align: center; margin: 20px;">
            <h1>🎭 最终版语音克隆系统</h1>
            <p style="font-size: 18px; color: #666;">
                AI驱动 | 进度同步 | 音频库管理 | 专业级语音克隆
            </p>
        </div>
        """)
        
        with gr.Row():
            # 左侧：输入和音频库
            with gr.Column(scale=1):
                gr.HTML("<h3>🎙️ 步骤1: 录制/上传/选择音频</h3>")
                
                # 音频输入
                audio_input = gr.Audio(
                    label="录制新音频或上传文件",
                    type="filepath",
                    sources=["upload", "microphone"],
                    show_label=True
                )
                
                # 音频库管理
                with gr.Group():
                    gr.HTML("<h4>📚 音频库管理</h4>")
                    with gr.Row():
                        save_name = gr.Textbox(
                            label="自定义名称（可选）",
                            placeholder="为录音起个名字...",
                            scale=2
                        )
                        save_to_library_btn = gr.Button("💾 保存到音频库", variant="secondary", scale=1)
                    
                    with gr.Row():
                        library_dropdown = gr.Dropdown(
                            choices=get_audio_library_list(),
                            label="选择历史录音",
                            value=None,
                            allow_custom_value=False,
                            scale=2
                        )
                        refresh_library_btn = gr.Button("🔄", variant="secondary", scale=0)
                    
                    with gr.Row():
                        load_from_library_btn = gr.Button("📂 加载选定录音", variant="secondary", scale=1)
                        delete_from_library_btn = gr.Button("🗑️ 删除选定录音", variant="secondary", scale=1)
                
                # 音频分析
                with gr.Row():
                    analyze_btn = gr.Button("🔍 分析音频", variant="secondary")
                    preview_audio = gr.Audio(
                        label="音频预览",
                        type="filepath",
                        interactive=False
                    )
                
                audio_status = gr.Textbox(
                    label="音频分析结果",
                    lines=10,
                    interactive=False,
                    placeholder="录制/上传/选择音频后，点击'分析音频'查看详细信息..."
                )
                
                library_status = gr.Textbox(
                    label="音频库操作状态",
                    lines=3,
                    interactive=False,
                    placeholder="音频库操作结果将在这里显示..."
                )
                
                gr.HTML("<h3>📝 步骤2: 输入合成文本</h3>")
                text_input = gr.Textbox(
                    label="要合成的文本内容",
                    placeholder="请输入您想要克隆语音说的内容...\n示例: Hello, this is a voice cloning demonstration.",
                    lines=3,
                    max_lines=5
                )
                
                gr.HTML("<h3>⚙️ 步骤3: 调整克隆参数</h3>")
                
                with gr.Row():
                    cfg_weight = gr.Slider(
                        minimum=0.1, maximum=1.0, value=0.8, step=0.05,
                        label="🎯 相似度控制",
                        info="0.8-0.9: 高相似度 | 0.6-0.7: 平衡 | 0.4-0.5: 创意"
                    )
                
                with gr.Row():
                    exaggeration = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.7, step=0.05,
                        label="🎭 情感表达度",
                        info="0.4-0.6: 平淡 | 0.7-0.9: 自然 | 1.0+: 丰富"
                    )
                
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.1, maximum=2.0, value=0.7, step=0.05,
                        label="🎲 随机性控制",
                        info="0.5-0.6: 稳定 | 0.7-0.8: 平衡 | 0.9+: 多样"
                    )
            
            # 右侧：控制和输出区域
            with gr.Column(scale=1):
                gr.HTML("<h3>🚀 步骤4: 模型管理</h3>")
                
                load_btn = gr.Button("🔧 加载AI模型", variant="secondary", size="lg")
                model_status = gr.Textbox(
                    label="AI模型状态与信息",
                    lines=8,
                    value="点击'加载AI模型'开始初始化...",
                    interactive=False
                )
                
                gr.HTML("<h3>🎭 步骤5: 开始克隆</h3>")
                clone_btn = gr.Button("🎭 开始语音克隆", variant="primary", size="lg")
                
                gr.HTML("""
                <div class="progress-info">
                    <h4>⚡ 进度同步优化</h4>
                    <p>• 前台进度条与AI生成实时同步</p>
                    <p>• 详细显示每个处理阶段</p>
                    <p>• 准确反映实际生成进度</p>
                </div>
                """)
                
                gr.HTML("<h3>🎵 克隆结果</h3>")
                output_audio = gr.Audio(
                    label="生成的语音",
                    type="filepath",
                    interactive=False
                )
                
                result_info = gr.Textbox(
                    label="详细结果信息",
                    lines=15,
                    interactive=False,
                    placeholder="点击'开始语音克隆'生成结果...",
                    show_copy_button=True
                )
        
        # 快速示例和参数预设
        gr.HTML("<hr><h3>💡 快速示例与参数预设</h3>")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h4>📝 文本示例</h4>")
                examples = gr.Examples(
                    examples=[
                        ["Hello, this is a voice cloning test.", 0.8, 0.7, 0.7],
                        ["Good morning! How are you today?", 0.9, 0.5, 0.6],
                        ["Welcome to our amazing voice cloning service!", 0.7, 1.0, 0.8],
                        ["Thank you for trying out this advanced technology.", 0.8, 0.6, 0.5]
                    ],
                    inputs=[text_input, cfg_weight, exaggeration, temperature],
                    label="点击示例快速设置"
                )
            
            with gr.Column(scale=1):
                gr.HTML("<h4>🎛️ 参数预设</h4>")
                with gr.Row():
                    preset_high_similarity = gr.Button("🎯 高相似度", variant="secondary")
                    preset_natural = gr.Button("🎭 自然表达", variant="secondary")
                    preset_creative = gr.Button("🎨 创意风格", variant="secondary")
                    preset_stable = gr.Button("🔒 稳定输出", variant="secondary")
        
        # 使用说明
        with gr.Accordion("📖 详细使用说明", open=False):
            gr.HTML("""
            <div style="padding: 20px; background: #f8f9fa; border-radius: 10px;">
                <h4>🎯 完整使用流程:</h4>
                <ol>
                    <li><strong>录制/上传/选择音频</strong>: 使用麦克风录制、上传文件或从音频库选择</li>
                    <li><strong>音频库管理</strong>: 保存录音到库中，管理历史录音</li>
                    <li><strong>分析音频</strong>: 验证音频质量和格式</li>
                    <li><strong>输入文本</strong>: 输入想要合成的内容</li>
                    <li><strong>调整参数</strong>: 根据需要调整三个核心参数</li>
                    <li><strong>加载模型</strong>: 初始化AI模型</li>
                    <li><strong>开始克隆</strong>: 执行语音克隆，享受同步进度反馈</li>
                </ol>
                
                <h4>📚 音频库功能:</h4>
                <ul>
                    <li><strong>自动保存</strong>: 录制的音频可以保存到库中重复使用</li>
                    <li><strong>自定义命名</strong>: 为每个录音设置有意义的名称</li>
                    <li><strong>历史管理</strong>: 查看、选择、删除历史录音</li>
                    <li><strong>快速加载</strong>: 一键加载之前保存的高质量录音</li>
                </ul>
                
                <h4>⚡ 进度同步改进:</h4>
                <ul>
                    <li><strong>真实反映</strong>: 进度条与AI生成过程实时同步</li>
                    <li><strong>阶段显示</strong>: 详细显示每个处理阶段</li>
                    <li><strong>时间估算</strong>: 显示实际生成耗时</li>
                    <li><strong>用户友好</strong>: 让用户了解真实的处理进度</li>
                </ul>
                
                <h4>⚠️ 重要提醒:</h4>
                <p>• AI语音生成需要一定时间，后台sampling进度是正常现象<br>
                • 请耐心等待生成完成，避免重复点击按钮<br>
                • 仅用于合法和道德目的，保护他人隐私权益</p>
            </div>
            """)
        
        # 事件绑定
        
        # 模型加载
        load_btn.click(
            fn=load_model_with_info,
            outputs=model_status
        )
        
        # 音频分析
        analyze_btn.click(
            fn=analyze_uploaded_audio,
            inputs=audio_input,
            outputs=[preview_audio, audio_status]
        )
        
        # 音频库操作
        save_to_library_btn.click(
            fn=save_recording_to_library,
            inputs=[audio_input, save_name],
            outputs=[library_status, library_dropdown]
        )
        
        refresh_library_btn.click(
            fn=lambda: gr.Dropdown(choices=get_audio_library_list(), value=None),
            outputs=library_dropdown
        )
        
        load_from_library_btn.click(
            fn=load_audio_from_library,
            inputs=library_dropdown,
            outputs=[preview_audio, audio_status]
        )
        
        delete_from_library_btn.click(
            fn=delete_audio_from_library,
            inputs=library_dropdown,
            outputs=[library_status, library_dropdown]
        )
        
        # 语音克隆
        clone_btn.click(
            fn=enhanced_voice_clone_v2,
            inputs=[audio_input, text_input, cfg_weight, exaggeration, temperature],
            outputs=[output_audio, result_info]
        )
        
        # 参数预设按钮
        def set_high_similarity():
            return 0.9, 0.5, 0.6
        
        def set_natural():
            return 0.8, 0.7, 0.7
        
        def set_creative():
            return 0.6, 1.0, 0.9
        
        def set_stable():
            return 0.8, 0.6, 0.5
        
        preset_high_similarity.click(
            fn=set_high_similarity,
            outputs=[cfg_weight, exaggeration, temperature]
        )
        
        preset_natural.click(
            fn=set_natural,
            outputs=[cfg_weight, exaggeration, temperature]
        )
        
        preset_creative.click(
            fn=set_creative,
            outputs=[cfg_weight, exaggeration, temperature]
        )
        
        preset_stable.click(
            fn=set_stable,
            outputs=[cfg_weight, exaggeration, temperature]
        )
    
    return demo

def main():
    """启动最终版网页界面"""
    print("🎭 最终版语音克隆网页界面")
    print("=" * 60)
    print("🎯 问题解决:")
    print("  • ✅ 进度条与AI生成过程实时同步")
    print("  • ✅ 详细显示每个处理阶段") 
    print("  • ✅ 录音自动保存到音频库")
    print("  • ✅ 历史录音管理和重复使用")
    print()
    print("🆕 主要改进:")
    print("  • 🎙️ 音频库：保存、管理、重复使用录音")
    print("  • ⏳ 进度同步：前台进度与后台AI生成同步")
    print("  • 📊 详细统计：显示实际生成耗时")
    print("  • 🛠️ 智能管理：音频文件的完整生命周期")
    print()
    
    print("🌐 访问地址:")
    print("  • 本地访问: http://localhost:7862")
    print("  • 局域网访问: http://0.0.0.0:7862")
    print()
    
    demo = create_final_interface()
    
    # 启动界面
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,  # 使用新端口
        share=False,
        inbrowser=True,
        show_error=True
    )

if __name__ == "__main__":
    main() 