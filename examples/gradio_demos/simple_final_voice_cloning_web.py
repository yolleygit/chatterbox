#!/usr/bin/env python3
"""
简化版最终语音克隆网页界面
- 更清晰的页面布局
- 更人性化的操作流程
- 简化的功能展示
"""

import gradio as gr
import os
import sys
import time
import json
import datetime
import threading
import numpy as np
import librosa
import torch
import torchaudio as ta
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from chatterbox import ChatterboxTTS
except ImportError:
    print("❌ 导入失败: 无法导入ChatterboxTTS")
    print("💡 请确保正确安装了chatterbox包")
    sys.exit(1)

# 全局变量
model = None
model_info = {}

# 创建输出目录
output_dir = Path("output/simple_voice_cloning")
audio_library_dir = Path("output/audio_library")
temp_dir = Path("output/temp")

for dir_path in [output_dir, audio_library_dir, temp_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)

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

def get_audio_library_list():
    """获取音频库简化列表"""
    index = load_audio_index()
    if not index:
        return ["选择历史录音..."]
    
    # 按创建时间倒序排列，只显示最近的10个
    sorted_items = sorted(index.items(), key=lambda x: x[1]['created_time'], reverse=True)[:10]
    audio_list = [f"{item['name']} ({item['duration']:.1f}s)" for _, item in sorted_items]
    return ["选择历史录音..."] + audio_list

def process_audio_input(audio_input, save_to_library=False, custom_name=""):
    """处理音频输入（录制/上传）"""
    if audio_input is None:
        return None, "🎙️ 请录制音频或上传文件", None
    
    try:
        # 处理音频
        if isinstance(audio_input, tuple):
            sr, audio_data = audio_input
            audio_data = audio_data.astype(np.float32)
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            duration = len(audio_data) / sr
            
            # 保存临时文件
            timestamp = int(time.time())
            temp_path = temp_dir / f"temp_audio_{timestamp}.wav"
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
            ta.save(temp_path, torch.from_numpy(audio_data).unsqueeze(0), sr)
            processed_path = str(temp_path)
        else:
            processed_path = audio_input
            audio_data, sr = librosa.load(audio_input, sr=None)
            duration = len(audio_data) / sr
        
        # 验证音频
        if duration < 1.0:
            return None, "❌ 音频太短，请录制至少1秒的音频", None
        if duration > 30.0:
            return None, "❌ 音频太长，请录制不超过30秒的音频", None
        
        # 生成分析信息
        max_amplitude = np.max(np.abs(audio_data))
        quality = "优秀" if 3 <= duration <= 10 and sr >= 16000 and max_amplitude > 0.1 else "可用"
        
        status_msg = f"""✅ 音频处理成功！

📊 音频信息:
• ⏱️ 时长: {duration:.1f}秒  
• 📈 采样率: {sr}Hz
• 🎵 质量: {quality}
• 💡 建议: {"音频质量很好，适合克隆" if quality == "优秀" else "可以使用，建议录制3-10秒高质量音频"}"""
        
        # 如果需要保存到音频库
        if save_to_library:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if custom_name.strip():
                safe_name = "".join(c for c in custom_name if c.isalnum() or c in (' ', '-', '_')).strip()
                filename = f"{safe_name}_{timestamp}.wav"
                display_name = safe_name
            else:
                filename = f"录音_{timestamp}.wav"
                display_name = f"录音_{timestamp}"
            
            # 保存到音频库
            library_path = audio_library_dir / filename
            if isinstance(audio_input, tuple):
                ta.save(library_path, torch.from_numpy(audio_data).unsqueeze(0), sr)
            else:
                import shutil
                shutil.copy2(processed_path, library_path)
            
            # 更新索引
            index = load_audio_index()
            index[filename] = {
                "filename": filename,
                "path": str(library_path),
                "created_time": timestamp,
                "name": display_name,
                "duration": duration,
                "sample_rate": sr
            }
            save_audio_index(index)
            
            status_msg += f"\n\n💾 已保存到音频库: {display_name}"
        
        return processed_path, status_msg, get_audio_library_list()
        
    except Exception as e:
        return None, f"❌ 音频处理失败: {str(e)}", None

def load_from_history(selected_audio):
    """从历史记录加载音频"""
    if not selected_audio or selected_audio == "选择历史录音...":
        return None, "请选择一个历史录音", None
    
    try:
        index = load_audio_index()
        for filename, info in index.items():
            display_name = f"{info['name']} ({info['duration']:.1f}s)"
            if display_name == selected_audio:
                if os.path.exists(info['path']):
                    status_msg = f"""📚 已加载历史录音: {info['name']}

📊 音频信息:
• ⏱️ 时长: {info['duration']:.1f}秒
• 📈 采样率: {info['sample_rate']}Hz  
• 📅 录制时间: {info['created_time']}

✅ 可以开始输入文本进行语音克隆！"""
                    return info['path'], status_msg, None
                else:
                    return None, f"❌ 音频文件不存在: {info['path']}", None
        
        return None, "❌ 未找到对应的音频文件", None
        
    except Exception as e:
        return None, f"❌ 加载失败: {str(e)}", None

def initialize_model():
    """初始化AI模型"""
    global model, model_info
    
    if model is None:
        apply_torch_load_patch()
        try:
            model = ChatterboxTTS.from_pretrained(device="cpu")
            device = "CPU (Apple Silicon)" if torch.backends.mps.is_available() else "CPU"
            model_info = {
                "name": "ChatterboxTTS",
                "device": device,
                "status": "已就绪"
            }
            return f"✅ AI模型加载成功！\n💻 运行设备: {device}\n🎭 状态: 已就绪"
        except Exception as e:
            return f"❌ 模型加载失败: {str(e)}\n💡 请检查网络连接和内存空间"
    else:
        return f"✅ AI模型已就绪！\n💻 设备: {model_info['device']}\n🎭 状态: {model_info['status']}"

def generate_voice_clone(audio_file, text, preset_type, progress=gr.Progress()):
    """生成语音克隆"""
    global model
    
    # 检查模型
    if model is None:
        return None, "❌ 请先等待AI模型加载完成！"
    
    # 检查输入
    if audio_file is None:
        return None, "❌ 请先录制或选择参考音频！"
    
    if not text.strip():
        return None, "❌ 请输入要合成的文本内容！"
    
    try:
        # 根据预设设置参数
        preset_params = {
            "标准模式": {"cfg_weight": 0.8, "exaggeration": 0.6, "temperature": 0.6},
            "自然表达": {"cfg_weight": 0.8, "exaggeration": 0.7, "temperature": 0.7},
            "创意风格": {"cfg_weight": 0.6, "exaggeration": 1.0, "temperature": 0.9}
        }
        
        params = preset_params.get(preset_type, preset_params["标准模式"])
        
        # 更新进度
        progress(0.1, desc="🔍 验证音频...")
        time.sleep(0.5)
        
        progress(0.2, desc="🎭 AI开始分析...")
        time.sleep(0.8)
        
        progress(0.4, desc="🧠 学习声音特征...")
        time.sleep(1.0)
        
        progress(0.6, desc="📝 处理文本内容...")
        time.sleep(0.8)
        
        progress(0.8, desc="🎵 生成语音...")
        
        # 生成音频
        start_time = time.time()
        result_audio = model.generate(
            text=text,
            audio_prompt_path=audio_file,
            cfg_weight=params["cfg_weight"],
            exaggeration=params["exaggeration"],
            temperature=params["temperature"]
        )
        
        progress(0.95, desc="💾 保存结果...")
        
        # 保存结果
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"cloned_voice_{timestamp}.wav"
        ta.save(output_path, result_audio.unsqueeze(0), 24000)
        
        progress(1.0, desc="✅ 生成完成！")
        
        generation_time = time.time() - start_time
        
        result_msg = f"""🎉 语音克隆成功！

📊 生成信息:
• 🎭 使用预设: {preset_type}
• ⏱️ 生成耗时: {generation_time:.1f}秒
• 📁 文件位置: {output_path.name}
• 🎵 采样率: 24kHz

💡 提示: 如果效果不理想，可以尝试其他预设模式或重新录制更清晰的参考音频"""
        
        return str(output_path), result_msg
        
    except Exception as e:
        return None, f"❌ 语音生成失败: {str(e)}\n💡 请检查音频文件和文本内容"

def create_simple_interface():
    """创建简化版网页界面"""
    
    # 简洁的CSS样式
    custom_css = """
    .gradio-container {
        max-width: 1000px !important;
        margin: auto !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .step-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        border: 2px solid #e0e6ed;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .step-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 25px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
    }
    
    .preset-btn {
        background: linear-gradient(45deg, #4CAF50, #45a049) !important;
        border: none !important;
        color: white !important;
        margin: 5px !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    .preset-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
    }
    
    .main-btn {
        background: linear-gradient(45deg, #FF6B6B, #FF8E53) !important;
        border: none !important;
        color: white !important;
        font-size: 18px !important;
        padding: 15px 30px !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }
    
    .main-btn:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3) !important;
    }
    """
    
    with gr.Blocks(css=custom_css, title="🎭 简化版语音克隆", theme=gr.themes.Soft()) as demo:
        
        # 标题区域
        gr.HTML("""
        <div style="text-align: center; margin: 30px 0;">
            <h1 style="color: #667eea; font-size: 36px; margin-bottom: 10px;">
                🎭 智能语音克隆系统
            </h1>
            <p style="font-size: 18px; color: #666; margin: 0;">
                简单三步 • 智能克隆 • 专业品质
            </p>
        </div>
        """)
        
        # 模型状态（始终显示）
        model_status = gr.Textbox(
            label="🤖 AI模型状态",
            value="🔄 正在初始化AI模型，请稍候...",
            interactive=False,
            lines=2
        )
        
        # 步骤1: 音频输入
        with gr.Group():
            gr.HTML('<div class="step-header">🎙️ 步骤 1: 录制或选择参考音频</div>')
            
            with gr.Row():
                # 音频录制/上传
                with gr.Column(scale=3):
                    audio_input = gr.Audio(
                        label="录制新音频或上传文件",
                        type="filepath",
                        sources=["upload", "microphone"],
                        show_label=True
                    )
                    
                    with gr.Row():
                        save_name = gr.Textbox(
                            label="给录音起个名字（可选）",
                            placeholder="比如：我的声音、工作录音...",
                            scale=2
                        )
                        save_btn = gr.Button("💾 保存", variant="secondary", scale=1)
                
                # 历史记录选择
                with gr.Column(scale=2):
                    gr.HTML("<p style='margin-bottom: 10px;'><strong>或者选择历史录音：</strong></p>")
                    history_dropdown = gr.Dropdown(
                        choices=get_audio_library_list(),
                        label="历史录音",
                        value=None,
                        scale=1
                    )
                    load_history_btn = gr.Button("📂 加载选中录音", variant="secondary")
            
            # 音频状态显示
            audio_status = gr.Textbox(
                label="📊 音频状态",
                lines=6,
                interactive=False,
                placeholder="请录制音频、上传文件或选择历史录音..."
            )
        
        # 步骤2: 文本和参数
        with gr.Group():
            gr.HTML('<div class="step-header">📝 步骤 2: 输入文本并选择风格</div>')
            
            text_input = gr.Textbox(
                label="要合成的文本内容",
                placeholder="请输入您希望克隆语音说的内容...\n例如：你好，这是一个语音克隆演示。",
                lines=4,
                max_lines=6
            )
            
            gr.HTML("<p style='margin: 15px 0 10px 0;'><strong>🎨 选择语音风格：</strong></p>")
            with gr.Row():
                preset_standard = gr.Button("🎯 标准模式", variant="secondary", elem_classes="preset-btn")
                preset_natural = gr.Button("🎭 自然表达", variant="secondary", elem_classes="preset-btn")
                preset_creative = gr.Button("🎨 创意风格", variant="secondary", elem_classes="preset-btn")
            
            selected_preset = gr.Textbox(
                label="当前选择",
                value="标准模式",
                interactive=False,
                lines=1
            )
        
        # 步骤3: 生成结果
        with gr.Group():
            gr.HTML('<div class="step-header">🎵 步骤 3: 生成语音克隆</div>')
            
            generate_btn = gr.Button(
                "🎭 开始语音克隆",
                variant="primary",
                size="lg",
                elem_classes="main-btn"
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    output_audio = gr.Audio(
                        label="🎵 生成的语音",
                        type="filepath",
                        interactive=False
                    )
                
                with gr.Column(scale=1):
                    result_info = gr.Textbox(
                        label="📊 生成结果",
                        lines=8,
                        interactive=False,
                        placeholder="点击'开始语音克隆'生成结果...",
                        show_copy_button=True
                    )
        
        # 使用说明（折叠面板）
        with gr.Accordion("📖 使用说明", open=False):
            gr.HTML("""
            <div style="padding: 20px; background: #f8f9fa; border-radius: 10px; line-height: 1.6;">
                <h4>🎯 快速上手指南：</h4>
                <ol style="padding-left: 20px;">
                    <li><strong>录制参考音频</strong>：使用麦克风录制3-10秒清晰音频，或上传音频文件</li>
                    <li><strong>输入合成文本</strong>：写下您希望克隆语音说的内容</li>
                    <li><strong>选择语音风格</strong>：
                        <ul style="margin: 5px 0; padding-left: 20px;">
                            <li>🎯 标准模式：稳定准确，适合正式场合</li>
                            <li>🎭 自然表达：自然流畅，适合日常对话</li>
                            <li>🎨 创意风格：富有表现力，适合创意内容</li>
                        </ul>
                    </li>
                    <li><strong>生成语音</strong>：点击按钮，等待AI生成结果</li>
                </ol>
                
                <h4>💡 获得最佳效果的技巧：</h4>
                <ul style="padding-left: 20px;">
                    <li>🎙️ 在安静环境中录制，避免背景噪音</li>
                    <li>⏱️ 录制3-10秒音频，时长适中效果最好</li>
                    <li>🗣️ 清晰发音，语速适中</li>
                    <li>💾 常用的声音可以保存到历史记录中重复使用</li>
                </ul>
                
                <h4>⚠️ 使用提醒：</h4>
                <p style="color: #666;">
                仅用于合法和道德目的，请尊重他人隐私权益。AI生成需要一定时间，请耐心等待。
                </p>
            </div>
            """)
        
        # 事件绑定
        
        # 页面加载时自动初始化模型
        demo.load(
            fn=initialize_model,
            outputs=model_status
        )
        
        # 音频处理
        def process_and_update_dropdown(audio_input):
            """处理音频并更新下拉框"""
            processed_path, status_msg, updated_choices = process_audio_input(audio_input, False, "")
            return processed_path, status_msg, gr.Dropdown(choices=updated_choices, value=None)
        
        audio_input.change(
            fn=process_and_update_dropdown,
            inputs=audio_input,
            outputs=[audio_input, audio_status, history_dropdown]
        )
        
        # 保存录音  
        def save_and_update_simple(audio_input, save_name):
            """保存录音并更新下拉框"""
            processed_path, status_msg, updated_choices = process_audio_input(audio_input, True, save_name)
            return processed_path, status_msg, gr.Dropdown(choices=updated_choices, value=None)
        
        save_btn.click(
            fn=save_and_update_simple,
            inputs=[audio_input, save_name],
            outputs=[audio_input, audio_status, history_dropdown]
        )
        
        # 加载历史录音
        def load_and_reset_dropdown(selected_audio):
            """加载历史录音并重置下拉框选择"""
            audio_path, status_msg, _ = load_from_history(selected_audio)
            return audio_path, status_msg, gr.Dropdown(value=None)
        
        load_history_btn.click(
            fn=load_and_reset_dropdown,
            inputs=history_dropdown,
            outputs=[audio_input, audio_status, history_dropdown]
        )
        
        # 预设选择
        preset_standard.click(
            fn=lambda: "标准模式",
            outputs=selected_preset
        )
        
        preset_natural.click(
            fn=lambda: "自然表达",
            outputs=selected_preset
        )
        
        preset_creative.click(
            fn=lambda: "创意风格",
            outputs=selected_preset
        )
        
        # 语音生成
        generate_btn.click(
            fn=generate_voice_clone,
            inputs=[audio_input, text_input, selected_preset],
            outputs=[output_audio, result_info]
        )
    
    return demo

def main():
    """启动简化版网页界面"""
    print("🎭 简化版语音克隆网页界面")
    print("=" * 50)
    print("🌟 特点:")
    print("  • 🎯 简洁清晰的三步流程")
    print("  • 🎨 美观的卡片式布局")
    print("  • 🚀 智能预设，操作简单")
    print("  • 📱 人性化的用户体验")
    print()
    print("🌐 访问地址:")
    print("  • 本地访问: http://localhost:7863")
    print("  • 局域网访问: http://0.0.0.0:7863")
    print()
    
    demo = create_simple_interface()
    
    # 启动界面
    demo.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=False,
        inbrowser=True,
        show_error=True
    )

if __name__ == "__main__":
    main() 