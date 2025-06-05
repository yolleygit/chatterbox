#!/usr/bin/env python3
"""
语音克隆网页界面
基于Gradio实现的完整语音克隆流程
包含：录音 → 参数调整 → 语音克隆 → 结果输出
"""

import gradio as gr
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import os
import tempfile
import time
from pathlib import Path
import numpy as np
import librosa

# 全局变量
model = None
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

def load_tts_model():
    """加载TTS模型"""
    global model
    if model is None:
        apply_torch_load_patch()
        try:
            model = ChatterboxTTS.from_pretrained(device="cpu")
            return "✅ 模型加载成功！"
        except Exception as e:
            return f"❌ 模型加载失败: {str(e)}"
    return "✅ 模型已就绪！"

def analyze_audio(audio_file):
    """分析上传的音频文件"""
    if audio_file is None:
        return "请先录制或上传音频文件"
    
    try:
        # 使用librosa分析音频
        if isinstance(audio_file, tuple):
            # Gradio麦克风录音格式 (sample_rate, audio_data)
            sr, audio_data = audio_file
            audio_data = audio_data.astype(np.float32)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)  # 转为单声道
            duration = len(audio_data) / sr
        else:
            # 上传的文件
            audio_data, sr = librosa.load(audio_file, sr=None)
            duration = len(audio_data) / sr
        
        # 分析结果
        max_amplitude = np.max(np.abs(audio_data))
        
        analysis = f"""
📊 音频分析结果:
• ⏱️ 时长: {duration:.2f}秒
• 📈 采样率: {sr}Hz
• 🔊 最大音量: {max_amplitude:.3f}
• 🎵 质量评估: {"优秀" if 3 <= duration <= 10 and sr >= 16000 else "可用"}

💡 建议:
• 最佳时长: 3-10秒
• 推荐采样率: 16kHz+
• 录音环境: 安静无噪音
"""
        return analysis
        
    except Exception as e:
        return f"❌ 音频分析失败: {str(e)}"

def clone_voice(reference_audio, text, cfg_weight, exaggeration, temperature, progress=gr.Progress()):
    """执行语音克隆"""
    global model
    
    if model is None:
        return None, "❌ 请先加载模型！"
    
    if reference_audio is None:
        return None, "❌ 请先录制或上传参考音频！"
    
    if not text.strip():
        return None, "❌ 请输入要合成的文本！"
    
    try:
        progress(0.1, desc="准备音频文件...")
        
        # 处理参考音频
        reference_path = None
        if isinstance(reference_audio, tuple):
            # 麦克风录音
            sr, audio_data = reference_audio
            audio_data = audio_data.astype(np.float32)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # 保存临时文件
            reference_path = temp_dir / f"reference_{int(time.time())}.wav"
            ta.save(reference_path, torch.from_numpy(audio_data).unsqueeze(0), sr)
        else:
            # 上传的文件
            reference_path = reference_audio
        
        progress(0.3, desc="开始语音克隆...")
        
        # 语音克隆
        wav_tensor = model.generate(
            text=text,
            audio_prompt_path=str(reference_path),
            cfg_weight=cfg_weight,
            exaggeration=exaggeration,
            temperature=temperature
        )
        
        progress(0.8, desc="保存结果...")
        
        # 保存结果
        timestamp = int(time.time())
        output_path = temp_dir / f"cloned_voice_{timestamp}.wav"
        ta.save(output_path, wav_tensor, model.sr)
        
        # 计算音频信息
        audio_duration = wav_tensor.shape[1] / model.sr
        
        progress(1.0, desc="完成！")
        
        result_info = f"""
🎉 语音克隆完成！

📊 生成信息:
• 🎼 音频时长: {audio_duration:.2f}秒
• 📁 保存路径: {output_path}
• ⚙️ 使用参数:
  - cfg_weight: {cfg_weight}
  - exaggeration: {exaggeration}
  - temperature: {temperature}

💡 提示: 可以调整参数重新生成以获得不同效果
"""
        
        return str(output_path), result_info
        
    except Exception as e:
        return None, f"❌ 语音克隆失败: {str(e)}"

def get_parameter_tips(cfg_weight, exaggeration, temperature):
    """获取参数调优提示"""
    tips = f"""
🎛️ 当前参数设置:

📊 cfg_weight = {cfg_weight}
{'🔴 低相似度 - 更有创意' if cfg_weight < 0.6 else '🟡 中等相似度 - 平衡' if cfg_weight < 0.8 else '🟢 高相似度 - 更像原声'}

🎭 exaggeration = {exaggeration}
{'🔵 低表达 - 平淡风格' if exaggeration < 0.6 else '🟡 中等表达 - 自然风格' if exaggeration < 1.0 else '🟠 高表达 - 夸张风格'}

🎲 temperature = {temperature}
{'🟢 低随机性 - 稳定一致' if temperature < 0.7 else '🟡 中等随机性 - 平衡' if temperature < 1.0 else '🔴 高随机性 - 多样变化'}

💡 优化建议:
• 想要更像原声 → 提高 cfg_weight
• 想要更有表现力 → 提高 exaggeration
• 想要更稳定 → 降低 temperature
• 想要更多变化 → 提高 temperature
"""
    return tips

def create_web_interface():
    """创建网页界面"""
    
    # 自定义CSS样式
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .tab-nav {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }
    .gr-button {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        border: none !important;
        color: white !important;
    }
    .gr-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
    }
    """
    
    with gr.Blocks(css=css, title="🎭 语音克隆工具", theme=gr.themes.Soft()) as interface:
        
        gr.HTML("""
        <div style="text-align: center; margin: 20px;">
            <h1>🎭 智能语音克隆系统</h1>
            <p style="font-size: 18px; color: #666;">
                通过AI技术实现个性化语音克隆，简单易用，效果出色
            </p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h2>🎙️ 步骤1: 准备参考音频</h2>")
                
                with gr.Tab("📱 麦克风录音"):
                    microphone_input = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="点击录制您的声音 (3-10秒)",
                        show_label=True
                    )
                    
                with gr.Tab("📁 上传文件"):
                    file_input = gr.Audio(
                        sources=["upload"],
                        type="filepath",
                        label="上传音频文件 (WAV/MP3/FLAC)",
                        show_label=True
                    )
                
                analyze_btn = gr.Button("🔍 分析音频质量", variant="secondary")
                audio_analysis = gr.Textbox(
                    label="音频分析结果",
                    lines=8,
                    interactive=False,
                    placeholder="点击'分析音频质量'查看详细信息..."
                )
        
            with gr.Column(scale=1):
                gr.HTML("<h2>⚙️ 步骤2: 调整参数</h2>")
                
                cfg_weight = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.8,
                    step=0.1,
                    label="🎯 相似度控制 (cfg_weight)",
                    info="越高越像参考音频"
                )
                
                exaggeration = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="🎭 情感表达度 (exaggeration)",
                    info="控制语音的表现力"
                )
                
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="🎲 随机性控制 (temperature)",
                    info="控制生成的多样性"
                )
                
                parameter_tips = gr.Textbox(
                    label="参数调优提示",
                    lines=8,
                    interactive=False,
                    value=get_parameter_tips(0.8, 0.7, 0.7)
                )
        
        gr.HTML("<hr>")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML("<h2>📝 步骤3: 输入文本</h2>")
                text_input = gr.Textbox(
                    label="要合成的文本内容",
                    placeholder="请输入您想要克隆语音说的内容...",
                    lines=3,
                    max_lines=5
                )
                
                with gr.Row():
                    load_model_btn = gr.Button("🚀 加载模型", variant="secondary")
                    clone_btn = gr.Button("🎭 开始语音克隆", variant="primary", size="lg")
                
                model_status = gr.Textbox(
                    label="模型状态",
                    value="点击'加载模型'开始",
                    interactive=False
                )
            
            with gr.Column(scale=1):
                gr.HTML("<h2>🎵 步骤4: 克隆结果</h2>")
                
                output_audio = gr.Audio(
                    label="克隆的语音",
                    type="filepath",
                    interactive=False
                )
                
                result_info = gr.Textbox(
                    label="生成结果信息",
                    lines=8,
                    interactive=False,
                    placeholder="点击'开始语音克隆'生成结果..."
                )
        
        # 预设示例
        gr.HTML("<hr>")
        gr.HTML("<h2>💡 快速开始示例</h2>")
        
        with gr.Row():
            gr.Examples(
                examples=[
                    ["Hello, this is a voice cloning demonstration.", 0.8, 0.7, 0.7],
                    ["Good morning! How are you doing today?", 0.9, 0.5, 0.6],
                    ["Welcome to our amazing voice cloning service!", 0.7, 1.0, 0.8],
                    ["Thank you for trying out this technology.", 0.8, 0.6, 0.5]
                ],
                inputs=[text_input, cfg_weight, exaggeration, temperature],
                label="点击示例快速填入参数"
            )
        
        # 使用说明
        with gr.Accordion("📖 使用说明", open=False):
            gr.HTML("""
            <div style="padding: 20px; background: #f8f9fa; border-radius: 10px;">
                <h3>🎯 使用步骤:</h3>
                <ol>
                    <li><strong>录制/上传音频</strong>: 使用麦克风录制3-10秒清晰语音，或上传音频文件</li>
                    <li><strong>分析音频</strong>: 点击"分析音频质量"检查音频是否符合要求</li>
                    <li><strong>调整参数</strong>: 根据需要调整相似度、表达度和随机性参数</li>
                    <li><strong>加载模型</strong>: 点击"加载模型"初始化AI模型</li>
                    <li><strong>输入文本</strong>: 输入您想要克隆语音说的内容</li>
                    <li><strong>开始克隆</strong>: 点击"开始语音克隆"生成结果</li>
                </ol>
                
                <h3>🎛️ 参数说明:</h3>
                <ul>
                    <li><strong>相似度控制</strong>: 0.8-0.9适合高质量克隆，0.5-0.7适合创意表达</li>
                    <li><strong>情感表达度</strong>: 0.4-0.6适合正式场合，0.7-1.2适合生动表达</li>
                    <li><strong>随机性控制</strong>: 0.5-0.7适合稳定输出，0.8-1.2适合多样变化</li>
                </ul>
                
                <h3>⚠️ 注意事项:</h3>
                <ul>
                    <li>仅用于合法和道德目的</li>
                    <li>使用他人声音需获得同意</li>
                    <li>不得用于欺诈或误导</li>
                    <li>首次加载模型需要较长时间</li>
                </ul>
            </div>
            """)
        
        # 事件绑定
        load_model_btn.click(
            fn=load_tts_model,
            outputs=model_status
        )
        
        analyze_btn.click(
            fn=analyze_audio,
            inputs=[microphone_input],  # 优先使用麦克风输入
            outputs=audio_analysis
        )
        
        # 当参数变化时更新提示
        for param in [cfg_weight, exaggeration, temperature]:
            param.change(
                fn=get_parameter_tips,
                inputs=[cfg_weight, exaggeration, temperature],
                outputs=parameter_tips
            )
        
        clone_btn.click(
            fn=clone_voice,
            inputs=[
                microphone_input,  # 使用麦克风输入作为主要参考
                text_input,
                cfg_weight,
                exaggeration,
                temperature
            ],
            outputs=[output_audio, result_info]
        )
        
        # 如果麦克风没有输入，尝试使用文件输入
        def use_file_input_if_needed(mic_input, file_input, text, cfg, exag, temp):
            reference = mic_input if mic_input is not None else file_input
            return clone_voice(reference, text, cfg, exag, temp)
        
        # 添加备用克隆按钮逻辑
        clone_btn.click(
            fn=use_file_input_if_needed,
            inputs=[microphone_input, file_input, text_input, cfg_weight, exaggeration, temperature],
            outputs=[output_audio, result_info]
        )
    
    return interface

def main():
    """启动网页应用"""
    print("🚀 启动语音克隆网页界面...")
    
    interface = create_web_interface()
    
    # 启动界面
    interface.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,        # 端口号
        share=False,             # 不创建公共链接
        debug=True,              # 调试模式
        show_error=True,         # 显示错误
        inbrowser=True           # 自动打开浏览器
    )

if __name__ == "__main__":
    main() 