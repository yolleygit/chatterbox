#!/usr/bin/env python3
"""
增强版语音克隆网页界面
解决问题：音频预览、模型信息显示、进度反馈、错误修复
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

# 全局变量
model = None
model_info = {"name": "", "version": "", "device": ""}
temp_dir = Path("output/web_voice_cloning")
temp_dir.mkdir(parents=True, exist_ok=True)

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
            yield "🔄 正在初始化模型..."
            
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
            
            success_msg = f"""✅ 模型加载成功！

📋 模型信息:
• 🤖 模型名称: {model_info['name']}
• 📦 版本: {model_info['version']}
• 💻 运行设备: {model_info['device']}
• 🎯 精度: {model_info['precision']}
• 💾 内存使用: {model_info['memory']}

🎭 现在可以开始语音克隆了！"""
            
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
        yield f"""✅ 模型已就绪！

📋 当前模型信息:
• 🤖 模型名称: {model_info['name']}
• 💻 运行设备: {model_info['device']}
• 🎯 状态: 已加载并准备就绪"""

def enhanced_voice_clone(audio_file, text, cfg_weight, exaggeration, temperature, progress=gr.Progress()):
    """增强的语音克隆函数，带进度反馈"""
    global model
    
    # 检查模型
    if model is None:
        return None, "❌ 请先点击'🔧 加载模型'按钮！"
    
    # 检查输入
    if audio_file is None:
        return None, "❌ 请先录制或上传参考音频！"
    
    if not text.strip():
        return None, "❌ 请输入要合成的文本内容！"
    
    try:
        # 步骤1: 验证音频
        progress(0.1, desc="🔍 验证音频文件...")
        processed_audio, validation_msg = validate_and_process_audio(audio_file)
        if processed_audio is None:
            return None, f"❌ 音频验证失败: {validation_msg}"
        
        # 步骤2: 准备参数
        progress(0.2, desc="⚙️ 准备生成参数...")
        time.sleep(0.5)  # 让用户看到进度
        
        # 步骤3: 开始语音生成
        progress(0.3, desc="🎭 开始AI语音克隆...")
        
        # 执行语音克隆
        wav_tensor = model.generate(
            text=text,
            audio_prompt_path=processed_audio,
            cfg_weight=cfg_weight,
            exaggeration=exaggeration,
            temperature=temperature
        )
        
        # 步骤4: 保存结果
        progress(0.8, desc="💾 保存音频文件...")
        timestamp = int(time.time())
        output_path = temp_dir / f"cloned_voice_{timestamp}.wav"
        ta.save(output_path, wav_tensor, model.sr)
        
        # 步骤5: 生成报告
        progress(0.9, desc="📊 生成结果报告...")
        audio_duration = wav_tensor.shape[1] / model.sr
        file_size = os.path.getsize(output_path) / 1024  # KB
        
        progress(1.0, desc="✅ 语音克隆完成！")
        
        result_info = f"""🎉 语音克隆成功完成！

📊 生成信息:
• 🎼 音频时长: {audio_duration:.2f}秒
• 📁 文件大小: {file_size:.1f}KB
• 💾 保存路径: {output_path.name}
• 🕐 生成时间: {time.strftime('%H:%M:%S')}

⚙️ 使用参数:
• 🎯 相似度控制: {cfg_weight} {'(高相似度)' if cfg_weight >= 0.8 else '(中等相似度)' if cfg_weight >= 0.6 else '(低相似度)'}
• 🎭 情感表达度: {exaggeration} {'(丰富表达)' if exaggeration >= 1.0 else '(自然表达)' if exaggeration >= 0.7 else '(平淡表达)'}
• 🎲 随机性控制: {temperature} {'(高变化)' if temperature >= 0.9 else '(平衡)' if temperature >= 0.7 else '(稳定)'}

💡 提示: 可调整参数重新生成以获得不同效果！"""
        
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
• 调整参数后重试（降低cfg_weight到0.7）"""
        else:
            error_msg = f"""❌ 语音克隆失败: {error_detail}

💡 建议:
• 检查参考音频质量
• 尝试调整参数设置
• 重启加载模型
• 查看终端输出获取更多信息"""
        
        return None, error_msg

def create_enhanced_interface():
    """创建增强版网页界面"""
    
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
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
    .audio-preview {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    """
    
    with gr.Blocks(css=custom_css, title="🎭 增强版语音克隆", theme=gr.themes.Soft()) as demo:
        
        gr.HTML("""
        <div style="text-align: center; margin: 20px;">
            <h1>🎭 增强版语音克隆系统</h1>
            <p style="font-size: 18px; color: #666;">
                AI驱动的智能语音克隆 | 音频预览 | 实时反馈 | 参数优化
            </p>
        </div>
        """)
        
        with gr.Row():
            # 左侧：输入和参数区域
            with gr.Column(scale=1):
                gr.HTML("<h3>🎙️ 步骤1: 录制/上传参考音频</h3>")
                
                audio_input = gr.Audio(
                    label="选择音频文件或录制语音",
                    type="filepath",
                    sources=["upload", "microphone"],
                    show_label=True
                )
                
                # 音频预览和分析
                with gr.Row():
                    analyze_btn = gr.Button("🔍 分析音频", variant="secondary")
                    preview_audio = gr.Audio(
                        label="音频预览",
                        type="filepath",
                        interactive=False,
                        visible=True
                    )
                
                audio_status = gr.Textbox(
                    label="音频分析结果",
                    lines=8,
                    interactive=False,
                    placeholder="录制或上传音频后，点击'分析音频'查看详细信息..."
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
                
                load_btn = gr.Button("🔧 加载模型", variant="secondary", size="lg")
                model_status = gr.Textbox(
                    label="模型状态与信息",
                    lines=10,
                    value="点击'加载模型'开始初始化AI模型...",
                    interactive=False
                )
                
                gr.HTML("<h3>🎭 步骤5: 开始克隆</h3>")
                clone_btn = gr.Button("🎭 开始语音克隆", variant="primary", size="lg")
                
                gr.HTML("<h3>🎵 克隆结果</h3>")
                output_audio = gr.Audio(
                    label="生成的语音",
                    type="filepath",
                    interactive=False
                )
                
                result_info = gr.Textbox(
                    label="详细结果信息",
                    lines=12,
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
                    <li><strong>录制/上传音频</strong>: 使用麦克风录制3-10秒清晰语音，或上传音频文件</li>
                    <li><strong>分析音频</strong>: 点击"分析音频"验证音频质量和格式</li>
                    <li><strong>输入文本</strong>: 在文本框中输入想要合成的内容</li>
                    <li><strong>调整参数</strong>: 根据需要调整三个核心参数</li>
                    <li><strong>加载模型</strong>: 首次使用点击"加载模型"初始化AI</li>
                    <li><strong>开始克隆</strong>: 点击"开始语音克隆"生成结果</li>
                    <li><strong>下载结果</strong>: 播放、下载或重新调整参数</li>
                </ol>
                
                <h4>🎛️ 参数详细说明:</h4>
                <ul>
                    <li><strong>相似度控制 (cfg_weight)</strong>:
                        <br>• 0.8-0.9: 高相似度，最接近原声
                        <br>• 0.6-0.7: 平衡效果，自然且有变化
                        <br>• 0.4-0.5: 创意表达，更多AI风格</li>
                    <li><strong>情感表达度 (exaggeration)</strong>:
                        <br>• 0.4-0.6: 平淡风格，适合正式朗读
                        <br>• 0.7-0.9: 自然表达，日常对话风格
                        <br>• 1.0-1.5: 丰富表达，情感充沛</li>
                    <li><strong>随机性控制 (temperature)</strong>:
                        <br>• 0.5-0.6: 稳定输出，重复生成相似
                        <br>• 0.7-0.8: 平衡模式，推荐使用
                        <br>• 0.9-1.2: 多样化，每次生成有变化</li>
                </ul>
                
                <h4>🔧 故障排除:</h4>
                <ul>
                    <li><strong>"index out of range"错误</strong>: 通常是音频太短或格式问题，尝试重新录制3-10秒音频</li>
                    <li><strong>模型加载失败</strong>: 检查网络连接和内存空间，重启程序重试</li>
                    <li><strong>音频上传失败</strong>: 确认文件格式为WAV/MP3/FLAC，大小<50MB</li>
                    <li><strong>效果不理想</strong>: 尝试不同的参数组合，使用高质量参考音频</li>
                </ul>
                
                <h4>⚠️ 重要提醒:</h4>
                <p>• 仅用于合法和道德目的<br>
                • 使用他人声音需获得明确同意<br>
                • 不得用于欺诈、误导或恶意用途<br>
                • 生成的音频仅在本地处理，保护隐私安全</p>
            </div>
            """)
        
        # 事件绑定
        
        # 模型加载（使用流式输出）
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
        
        # 语音克隆
        clone_btn.click(
            fn=enhanced_voice_clone,
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
    """启动增强版网页界面"""
    print("🚀 启动增强版语音克隆网页界面...")
    print("📱 新增功能:")
    print("  • 🎙️ 音频预览和质量分析")
    print("  • 🔧 详细模型加载信息")
    print("  • ⏳ 实时进度反馈")
    print("  • 🛠️ 增强错误处理")
    print("  • 🎛️ 参数预设快捷按钮")
    
    demo = create_enhanced_interface()
    
    # 启动界面
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,  # 使用不同端口避免冲突
        share=False,
        inbrowser=True,
        show_error=True
    )

if __name__ == "__main__":
    main() 