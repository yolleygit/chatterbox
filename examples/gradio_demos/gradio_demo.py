#!/usr/bin/env python3
"""
Chatterbox TTS Gradio Web界面演示
提供友好的Web界面进行语音合成
"""

import torch
import gradio as gr
from chatterbox.tts import ChatterboxTTS

# 全局模型实例
model = None

def load_model():
    """加载TTS模型"""
    global model
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"🔧 加载模型到设备: {device}")
        model = ChatterboxTTS.from_pretrained(device)
        print("✅ 模型加载完成")
    return model

def generate_speech(text, reference_audio, exaggeration, cfg_weight, temperature):
    """生成语音的主函数"""
    try:
        # 加载模型
        tts_model = load_model()
        
        # 生成语音
        if reference_audio is not None:
            # 使用参考音频进行语音克隆
            wav = tts_model.generate(
                text,
                audio_prompt_path=reference_audio,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature
            )
        else:
            # 使用默认声音
            wav = tts_model.generate(
                text,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature
            )
        
        # 返回音频数据 (采样率, 音频数组)
        return (tts_model.sr, wav.squeeze(0).numpy())
        
    except Exception as e:
        gr.Warning(f"生成失败: {str(e)}")
        return None

def create_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(title="Chatterbox TTS Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎤 Chatterbox TTS 演示")
        gr.Markdown("开源的高质量语音合成系统，支持零样本语音克隆和情感控制")
        
        with gr.Row():
            with gr.Column():
                # 输入控件
                text_input = gr.Textbox(
                    label="📝 输入文本",
                    placeholder="请输入要合成的文本...",
                    lines=3,
                    value="你好，欢迎使用Chatterbox语音合成系统！"
                )
                
                ref_audio = gr.Audio(
                    label="🎯 参考音频（可选）",
                    sources=["upload", "microphone"],
                    type="filepath"
                )
                
                # 参数控制
                with gr.Accordion("🎛️ 高级参数", open=False):
                    exaggeration = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.5,
                        step=0.1,
                        label="🎭 情感夸张度",
                        info="控制语音的表现力"
                    )
                    
                    cfg_weight = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        label="⚖️ 一致性权重",
                        info="控制生成的一致性"
                    )
                    
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.8,
                        step=0.1,
                        label="🌡️ 随机性",
                        info="控制语音的变化性"
                    )
                
                generate_btn = gr.Button("🎵 生成语音", variant="primary")
            
            with gr.Column():
                # 输出音频
                audio_output = gr.Audio(
                    label="🔊 生成的语音",
                    type="numpy"
                )
                
                # 使用说明
                gr.Markdown("""
                ### 📋 使用说明:
                1. **输入文本**: 在左侧文本框中输入要合成的内容
                2. **参考音频** (可选): 上传3-10秒的清晰语音作为声音参考
                3. **调节参数**: 根据需要调整情感强度和其他参数
                4. **生成语音**: 点击生成按钮获得合成语音
                
                ### 💡 参数建议:
                - **日常对话**: 夸张度0.5-0.7
                - **正式场合**: 夸张度0.3，一致性0.7
                - **表演朗读**: 夸张度1.0+，随机性0.9+
                """)
        
        # 绑定生成函数
        generate_btn.click(
            fn=generate_speech,
            inputs=[text_input, ref_audio, exaggeration, cfg_weight, temperature],
            outputs=audio_output
        )
    
    return demo

def main():
    """主函数"""
    print("🌐 启动Chatterbox TTS Web界面...")
    
    # 创建界面
    demo = create_interface()
    
    # 启动服务
    demo.launch(
        share=False,        # 设为True可生成公网链接
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

if __name__ == "__main__":
    main() 