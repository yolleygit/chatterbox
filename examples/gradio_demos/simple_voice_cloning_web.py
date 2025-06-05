#!/usr/bin/env python3
"""
简化版语音克隆网页界面
基本功能：上传音频 → 参数调整 → 语音克隆 → 结果输出
"""

import gradio as gr
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import os
import time
from pathlib import Path

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

def load_model():
    """加载TTS模型"""
    global model
    if model is None:
        apply_torch_load_patch()
        try:
            model = ChatterboxTTS.from_pretrained(device="cpu")
            return "✅ 模型加载成功！现在可以开始语音克隆了。"
        except Exception as e:
            return f"❌ 模型加载失败: {str(e)}"
    return "✅ 模型已就绪！"

def simple_voice_clone(audio_file, text, cfg_weight, exaggeration, temperature):
    """简化的语音克隆函数"""
    global model
    
    # 检查模型
    if model is None:
        return None, "❌ 请先点击'加载模型'按钮！"
    
    # 检查输入
    if audio_file is None:
        return None, "❌ 请上传参考音频文件！"
    
    if not text.strip():
        return None, "❌ 请输入要合成的文本！"
    
    try:
        # 语音克隆
        wav_tensor = model.generate(
            text=text,
            audio_prompt_path=audio_file,
            cfg_weight=cfg_weight,
            exaggeration=exaggeration,
            temperature=temperature
        )
        
        # 保存结果
        timestamp = int(time.time())
        output_path = temp_dir / f"cloned_voice_{timestamp}.wav"
        ta.save(output_path, wav_tensor, model.sr)
        
        # 计算信息
        audio_duration = wav_tensor.shape[1] / model.sr
        
        result_info = f"""🎉 语音克隆完成！

📊 生成信息:
• 🎼 音频时长: {audio_duration:.2f}秒
• 📁 保存路径: {output_path}
• ⚙️ 使用参数:
  - 相似度: {cfg_weight}
  - 表达度: {exaggeration}
  - 随机性: {temperature}

💡 可以调整参数重新生成获得不同效果"""
        
        return str(output_path), result_info
        
    except Exception as e:
        return None, f"❌ 语音克隆失败: {str(e)}"

def create_simple_interface():
    """创建简化的网页界面"""
    
    with gr.Blocks(title="🎭 语音克隆", theme=gr.themes.Soft()) as demo:
        
        gr.HTML("""
        <div style="text-align: center; margin: 20px;">
            <h1>🎭 语音克隆系统</h1>
            <p>使用AI技术实现个性化语音克隆</p>
        </div>
        """)
        
        with gr.Row():
            # 左侧：输入区域
            with gr.Column(scale=1):
                gr.HTML("<h3>📁 步骤1: 上传参考音频</h3>")
                audio_input = gr.Audio(
                    label="选择音频文件",
                    type="filepath",
                    sources=["upload", "microphone"]
                )
                
                gr.HTML("<h3>📝 步骤2: 输入文本</h3>")
                text_input = gr.Textbox(
                    label="要合成的文本内容",
                    placeholder="请输入您想要克隆语音说的内容...",
                    lines=3
                )
                
                gr.HTML("<h3>⚙️ 步骤3: 调整参数</h3>")
                
                cfg_weight = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.8, step=0.1,
                    label="🎯 相似度控制",
                    info="越高越像参考音频"
                )
                
                exaggeration = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                    label="🎭 情感表达度",
                    info="控制语音表现力"
                )
                
                temperature = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                    label="🎲 随机性控制",
                    info="控制生成多样性"
                )
            
            # 右侧：控制和输出区域
            with gr.Column(scale=1):
                gr.HTML("<h3>🚀 步骤4: 开始克隆</h3>")
                
                load_btn = gr.Button("🔧 加载模型", variant="secondary", size="lg")
                model_status = gr.Textbox(
                    label="模型状态",
                    value="点击'加载模型'开始",
                    interactive=False
                )
                
                clone_btn = gr.Button("🎭 开始语音克隆", variant="primary", size="lg")
                
                gr.HTML("<h3>🎵 克隆结果</h3>")
                output_audio = gr.Audio(label="生成的语音", type="filepath")
                
                result_info = gr.Textbox(
                    label="生成结果",
                    lines=8,
                    interactive=False,
                    placeholder="点击'开始语音克隆'查看结果..."
                )
        
        # 示例区域
        gr.HTML("<hr><h3>💡 快速示例</h3>")
        
        examples = gr.Examples(
            examples=[
                ["Hello, this is a voice cloning test.", 0.8, 0.7, 0.7],
                ["Good morning! How are you today?", 0.9, 0.5, 0.6],
                ["Welcome to our voice cloning service!", 0.7, 1.0, 0.8],
                ["Thank you for trying this technology.", 0.8, 0.6, 0.5]
            ],
            inputs=[text_input, cfg_weight, exaggeration, temperature],
            label="点击示例快速设置参数"
        )
        
        # 使用说明
        with gr.Accordion("📖 使用说明", open=False):
            gr.HTML("""
            <div style="padding: 15px; background: #f8f9fa; border-radius: 8px;">
                <h4>🎯 使用步骤:</h4>
                <ol>
                    <li><strong>上传音频</strong>: 选择3-10秒的清晰音频文件</li>
                    <li><strong>输入文本</strong>: 输入想要合成的内容</li>
                    <li><strong>调整参数</strong>: 根据需要调整各项参数</li>
                    <li><strong>加载模型</strong>: 首次使用需要加载AI模型</li>
                    <li><strong>开始克隆</strong>: 点击按钮生成语音</li>
                </ol>
                
                <h4>🎛️ 参数说明:</h4>
                <ul>
                    <li><strong>相似度控制</strong>: 0.8-0.9 高质量克隆 | 0.5-0.7 创意表达</li>
                    <li><strong>情感表达度</strong>: 0.4-0.6 正式场合 | 0.7-1.2 生动表达</li>
                    <li><strong>随机性控制</strong>: 0.5-0.7 稳定输出 | 0.8-1.2 多样变化</li>
                </ul>
                
                <h4>⚠️ 注意事项:</h4>
                <p>仅用于合法目的，使用他人声音需获得同意，不得用于欺诈。</p>
            </div>
            """)
        
        # 事件绑定
        load_btn.click(
            fn=load_model,
            outputs=model_status
        )
        
        clone_btn.click(
            fn=simple_voice_clone,
            inputs=[audio_input, text_input, cfg_weight, exaggeration, temperature],
            outputs=[output_audio, result_info]
        )
    
    return demo

def main():
    """启动简化版网页界面"""
    print("🚀 启动简化版语音克隆网页界面...")
    
    demo = create_simple_interface()
    
    # 启动界面
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True
    )

if __name__ == "__main__":
    main() 