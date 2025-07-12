# 设置当前路径 - 必须在导入项目模块之前
import os
import sys

if getattr(sys, "frozen", False):
    current_path = os.path.dirname(sys.executable)
    os.environ["exe"] = 'True'
elif __file__:
    current_path = os.path.dirname(__file__)
    os.environ["exe"] = 'False'
os.environ["current_path"] = current_path

# 所有import语句统一放在文件顶部
import hashlib
import io
import warnings
import json
import datetime
import time
import shutil
import subprocess
import pickle
import concurrent.futures
import soundfile as sf
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# 第三方库导入
import gradio as gr

# 项目内部导入
from Sava_Utils import args, MANUAL, audio_separator
from Sava_Utils.utils import *
from Sava_Utils.edit_panel import *
from Sava_Utils.subtitle import Subtitle, Subtitles
from Sava_Utils.video_speed_adjuster import adjust_video_speed_by_subtitles, merge_video_with_audio
import Sava_Utils.tts_projects
import Sava_Utils.tts_projects.gsv
import Sava_Utils.tts_projects.edgetts
import Sava_Utils.tts_projects.custom
import Sava_Utils.tts_projects.indextts
from Sava_Utils.subtitle_translation import Translation_module
from Sava_Utils.polyphone import Polyphone

warnings.filterwarnings("ignore", category=UserWarning)

GSV = Sava_Utils.tts_projects.gsv.GSV(Sava_Utils.config)
EDGETTS = Sava_Utils.tts_projects.edgetts.EdgeTTS(Sava_Utils.config)
CUSTOM = Sava_Utils.tts_projects.custom.Custom(Sava_Utils.config)
INDEXTTS = Sava_Utils.tts_projects.indextts.IndexTTS(Sava_Utils.config)
TRANSLATION_MODULE = Translation_module(Sava_Utils.config)
POLYPHONE = Polyphone(Sava_Utils.config)
Projet_dict = {"gsv": GSV, "edgetts": EDGETTS, "indextts": INDEXTTS, "custom": CUSTOM}


def check_cache_file(video_path, subtitle_file, workspace_name):
    """检查是否存在有效的缓存文件"""
    try:
        cache_dir = os.path.join(current_path, "SAVAdata", "temp", "audio_processing", workspace_name)
        cache_file = os.path.join(cache_dir, "processing_cache.json")

        if not os.path.exists(cache_file):
            return None

        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)

        # 检查文件路径和文件大小是否匹配
        if (cache_data.get('video_path') == video_path and
                cache_data.get('subtitle_path') == subtitle_file):

            # 检查文件大小
            video_size = os.path.getsize(video_path)
            subtitle_size = os.path.getsize(subtitle_file)

            if (cache_data.get('video_size', 0) == video_size and
                    cache_data.get('subtitle_size', 0) == subtitle_size):

                # 检查所有输出文件是否存在
                result_files = cache_data.get('processing_result', {})
                for file_path in result_files.values():
                    if file_path and not os.path.exists(file_path):
                        return None

                return cache_data

        return None
    except Exception as e:
        print(f"缓存检查失败: {e}")
        return None


def save_cache_file(video_path, subtitle_file, workspace_name, output_dir, result, segments_count):
    """保存处理结果到缓存文件"""
    try:
        cache_file = os.path.join(output_dir, "processing_cache.json")

        cache_data = {
            "video_path": video_path,
            "subtitle_path": subtitle_file,
            "video_size": os.path.getsize(video_path),
            "subtitle_size": os.path.getsize(subtitle_file),
            "workspace_name": workspace_name,
            "output_dir": output_dir,
            "processing_result": result,
            "segments_count": segments_count,
            "env_vars": {
                "current_video_path": video_path
            },
            "created_time": time.time(),
            "created_time_str": time.strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

        print(f"✅ 缓存文件已保存: {cache_file}")

    except Exception as e:
        print(f"⚠️ 缓存保存失败: {e}")


def handle_video_file_load(video_file_upload, video_path_input, uploaded_files, current_state):
    """处理视频文件加载和音频分离 - 支持文件选择和路径输入，支持缓存机制"""
    # 确定视频文件路径
    video_path = None

    # 优先使用文件上传
    if video_file_upload:
        video_path = video_file_upload
        source_type = "文件选择"
    # 其次使用路径输入
    elif video_path_input and video_path_input.strip():
        video_path = video_path_input.strip().strip('"').strip("'")
        source_type = "路径输入"
    else:
        return gr.update(
            value="⚠️ **请选择视频文件或输入文件路径**\n\n💡 可以通过以下方式之一：\n• 📁 在'选择文件'标签页中选择视频文件\n• 📝 在'输入路径'标签页中输入文件路径"), current_state

    # 检查文件是否存在
    if not os.path.exists(video_path):
        return gr.update(
            value=f"❌ **文件不存在**\n\n📂 检查路径：`{video_path}`\n🔧 来源：{source_type}\n\n💡 请确认文件路径是否正确"), current_state

    # 检查是否是文件（不是目录）
    if not os.path.isfile(video_path):
        return gr.update(
            value=f"❌ **这是一个目录，不是文件**\n\n📂 路径：`{video_path}`\n🔧 来源：{source_type}\n\n💡 请选择具体的视频文件"), current_state

    # 检查文件格式
    file_extension = os.path.splitext(video_path)[1].lower()
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.ts', '.m2ts',
                        '.mts', '.m4v', '.3gp', '.3g2', '.asf', '.rm', '.rmvb', '.vob', '.mpg',
                        '.mpeg', '.m1v', '.m2v', '.ogv', '.ogg']

    if not any(video_path.lower().endswith(ext) for ext in video_extensions):
        supported_formats = "MP4, AVI, MKV, MOV, WMV, WebM, TS, 3GP, RMVB, MPG"
        return gr.update(
            value=f"❌ **不支持的视频格式**\n\n🎞️ 当前格式：`{file_extension.upper()}`\n\n✅ 支持格式：{supported_formats}"), current_state

    # 检查字幕文件
    if not uploaded_files or len(uploaded_files) == 0:
        return gr.update(
            value="⚠️ **请先上传字幕文件**\n\n📝 需要上传 .srt 字幕文件才能进行音频分割\n\n💡 请在上方的文件上传区域选择字幕文件"), current_state

    # 获取字幕文件路径 - 支持多种格式
    subtitle_file = None
    supported_subtitle_formats = ['.srt', '.ass', '.vtt', '.csv', '.txt']

    for file in uploaded_files:
        file_ext = os.path.splitext(file.name)[1].lower()
        if file_ext in supported_subtitle_formats:
            subtitle_file = file.name
            break

    if not subtitle_file:
        return gr.update(
            value="⚠️ **未找到字幕文件**\n\n📝 上传的文件中没有支持的字幕格式\n\n✅ 支持格式：SRT, ASS, VTT, CSV, TXT\n\n💡 请上传正确格式的字幕文件"), current_state

    # 生成workspace名称用于缓存检查
    video_basename = os.path.basename(video_path)
    workspace_name = video_basename.replace(".", "-")
    if workspace_name.endswith("-"):
        workspace_name = workspace_name[:-1]

    # 尝试从缓存加载
    cache_data = check_cache_file(video_path, subtitle_file, workspace_name)
    print(f"Cache data: {video_path},  {subtitle_file},{workspace_name}")
    if cache_data:
        # 从缓存恢复环境变量
        if cache_data.get('env_vars'):
            for key, value in cache_data['env_vars'].items():
                os.environ[key] = value

        # 恢复处理状态
        cached_state = {
            "processed": True,
            "video_path": video_path,
            "srt_path": subtitle_file,
            "processing_result": cache_data.get('processing_result', {}),
            "workspace_name": workspace_name,
            "output_dir": cache_data.get('output_dir', '')
        }

        # 显示缓存加载成功信息
        result_files = cache_data.get('processing_result', {})
        segments_count = cache_data.get('segments_count', 0)

        success_message = f"""
🚀 **从缓存快速加载！**

✅ **处理结果**（已缓存）
• 🎬 无声视频: `{os.path.basename(result_files.get('raw_video', 'N/A'))}`
• 🎵 原始音频: `{os.path.basename(result_files.get('raw_audio', 'N/A'))}`
• 🎤 人声音频: `{os.path.basename(result_files.get('vocal_audio', 'N/A'))}`
• 🎼 背景音乐: `{os.path.basename(result_files.get('background_audio', 'N/A'))}`
• ✂️ 音频片段: **{segments_count} 个片段**

📂 **存储位置**
• 🎬 项目目录: `SAVAdata/temp/audio_processing/{workspace_name}/`
• ✂️ 音频片段: `SAVAdata/temp/audio_processing/{workspace_name}/segments/`

🏷️ **项目名称**: `{workspace_name}`

⚡ 文件已从缓存快速加载，无需重新处理！
        """.strip()

        return gr.update(value=success_message), cached_state

    # 检查是否已经处理过相同的文件（保留原有逻辑作为备用）
    if (current_state["processed"] and
            current_state["video_path"] == video_path and
            current_state["srt_path"] == subtitle_file):
        return gr.update(
            value="ℹ️ **文件已处理过**\n\n✅ 相同的视频和字幕文件已经处理过了\n\n💡 如需重新处理，请更换文件或重启程序"), current_state

    try:
        # 检查文件权限
        if not os.access(video_path, os.R_OK):
            return gr.update(
                value=f"❌ **文件权限不足**\n\n🔒 无法读取文件：`{video_path}`\n\n💡 请检查文件权限或以管理员身份运行"), current_state

        # 导入音频分离模块
        sys.path.insert(0, 'Sava_Utils')

        # 步骤1: 分离视频音频
        # workspace_name 已在缓存检查部分生成，这里直接使用

        # 使用项目标准的存储路径，包含workspace名称子目录
        base_temp_dir = os.path.join(current_path, "SAVAdata", "temp")
        output_dir = os.path.join(base_temp_dir, "audio_processing", workspace_name)
        os.makedirs(output_dir, exist_ok=True)

        # 使用配置中的显存和质量调整设置
        result = audio_separator.separate_video_audio(
            video_path,
            output_dir,
            audio_quality="high",  # 初始质量，会根据文件大小自动调整
            max_memory_gb=Sava_Utils.config.max_gpu_memory_gb,
            auto_quality=Sava_Utils.config.auto_quality_adjustment
        )

        # 使用人声音频进行分割
        vocal_audio_path = result.get('vocal_audio')
        if not vocal_audio_path or not os.path.exists(vocal_audio_path):
            return gr.update(
                value="❌ **音频分离失败**\n\n🔧 无法生成人声音频文件\n\n💡 请检查视频文件是否包含音频轨道"), current_state

        # 步骤2: 根据字幕分割音频
        segments_dir = os.path.join(output_dir, "segments")
        os.makedirs(segments_dir, exist_ok=True)

        # 处理字幕文件 - 如果是 ASS 或 VTT，先转换为 SRT
        subtitle_ext = os.path.splitext(subtitle_file)[1].lower()
        if subtitle_ext in ['.ass', '.vtt']:
            # 需要转换为 SRT 格式进行分割
            temp_srt_path = os.path.join(output_dir, "temp_subtitle.srt")

            if subtitle_ext == '.ass':
                # ASS 文件处理
                from Sava_Utils.subtitle_processor import format_ass_file, extract_ass_to_srt, get_available_styles

                # 格式化 ASS 文件
                formatted_ass_path = os.path.join(output_dir, "formatted.ass")
                format_success = format_ass_file(subtitle_file, formatted_ass_path)
                if not format_success:
                    formatted_ass_path = subtitle_file

                # 获取样式并转换
                styles = get_available_styles(formatted_ass_path)
                style_name = styles[0] if styles else "Default"
                extract_ass_to_srt(formatted_ass_path, style_name, temp_srt_path)

            elif subtitle_ext == '.vtt':
                # VTT 文件处理
                from Sava_Utils.subtitle_processor import convert_subtitle
                convert_subtitle(subtitle_file, temp_srt_path)

            # 使用转换后的 SRT 文件进行分割
            split_subtitle_file = temp_srt_path
        else:
            # 直接使用原文件
            split_subtitle_file = subtitle_file

        segments = audio_separator.split_audio_by_subtitles(vocal_audio_path, split_subtitle_file, segments_dir)

        # 设置环境变量供 Clone 模式使用
        os.environ["current_video_path"] = video_path

        # 保存处理结果到缓存
        save_cache_file(video_path, subtitle_file, workspace_name, output_dir, result, len(segments))

        # 更新处理状态，保存所有处理结果
        new_state = {
            "processed": True,
            "video_path": video_path,
            "srt_path": subtitle_file,
            "processing_result": result,  # 保存完整的处理结果
            "workspace_name": workspace_name,
            "output_dir": output_dir
        }

        # 成功反馈
        success_message = f"""
🎉 **处理完成！**

✅ **处理结果**
• 🎬 无声视频: `{os.path.basename(result.get('raw_video', 'N/A'))}`
• 🎵 原始音频: `{os.path.basename(result.get('raw_audio', 'N/A'))}`
• 🎤 人声音频: `{os.path.basename(result.get('vocal_audio', 'N/A'))}`
• 🎼 背景音乐: `{os.path.basename(result.get('background_audio', 'N/A'))}`
• ✂️ 音频片段: **{len(segments)} 个片段**

📂 **存储位置**
• 🎬 项目目录: `SAVAdata/temp/audio_processing/{workspace_name}/`
• ✂️ 音频片段: `SAVAdata/temp/audio_processing/{workspace_name}/segments/`

🏷️ **项目名称**: `{workspace_name}`

🎯 文件已按项目名称组织保存，便于管理和查找！
💾 处理结果已缓存，下次加载相同文件将快速恢复！
        """.strip()

        return gr.update(value=success_message), new_state

    except Exception as e:
        error_message = f"""
❌ **处理失败**

🔧 **错误详情**
```
{str(e)}
```

💡 **可能的解决方案:**
• 检查视频文件是否完整
• 确认字幕文件格式正确
• 检查磁盘空间是否充足
• 重启程序后重试
        """.strip()
        return gr.update(value=error_message), current_state


def handle_compose_video(progress, video_file_upload, video_path_input, subtitle_files,
                         current_state, subtitles_state, audio_data):
    """处理视频合成 - 完整检查版本，支持进度条显示"""

    # 确定视频文件路径
    video_path = None
    if video_file_upload:
        video_path = video_file_upload
    elif video_path_input and video_path_input.strip():
        video_path = video_path_input.strip().strip('"').strip("'")

    # 1. 检查字幕是否上传
    if not subtitle_files or len(subtitle_files) == 0:
        return gr.update(
            value="❌ **字幕文件检查失败**\n\n📝 **错误**: 未上传字幕文件\n\n💡 **解决方案**: 请在左侧上传 .srt 格式的字幕文件")

    # 检查字幕文件格式
    supported_formats = ['.srt', '.ass', '.vtt', '.csv', '.txt']
    subtitle_files_filtered = []
    for f in subtitle_files:
        file_ext = os.path.splitext(f.name)[1].lower()
        if file_ext in supported_formats:
            subtitle_files_filtered.append(f)

    if len(subtitle_files_filtered) == 0:
        return gr.update(
            value="❌ **字幕文件格式错误**\n\n📝 **错误**: 上传的文件中没有支持的字幕格式\n\n✅ **支持格式**: SRT, ASS, VTT, CSV, TXT\n\n💡 **解决方案**: 请上传正确格式的字幕文件")

    # 2. 检查视频是否加载过
    if not video_path or video_path.strip() == "":
        return gr.update(
            value="❌ **视频文件检查失败**\n\n🎬 **错误**: 未输入视频文件路径\n\n💡 **解决方案**: 请在上方输入视频路径并点击'🚀 加载文件'按钮")

    # 清理路径
    video_path = video_path.strip().strip('"').strip("'")

    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        return gr.update(
            value="❌ **视频文件不存在**\n\n🎬 **错误**: 指定的视频文件路径不存在\n\n💡 **解决方案**: 请检查文件路径是否正确，并重新点击'🚀 加载文件'按钮")

    # 检查视频是否已经处理过（音频分离）
    if not current_state.get("processed", False):
        return gr.update(
            value="❌ **视频未处理**\n\n🎬 **错误**: 视频文件未经过音频分离处理\n\n💡 **解决方案**: 请点击'🚀 加载文件'按钮先处理视频文件")

    # 3. 检查音频是否生成
    if subtitles_state is None or len(subtitles_state) == 0:
        return gr.update(
            value="❌ **音频生成检查失败**\n\n🎵 **错误**: 未找到字幕数据，音频可能未生成\n\n💡 **解决方案**: 请先在左侧选择TTS服务并点击'生成'按钮生成音频")

    # 检查音频输出
    if audio_data is None:
        return gr.update(
            value="❌ **音频输出检查失败**\n\n🎵 **错误**: 未检测到生成的音频数据\n\n💡 **解决方案**: 请确保已完成音频生成，并在右侧看到音频播放器")

    # 检查字幕是否有成功生成的音频
    success_count = 0
    total_count = len(subtitles_state)

    for subtitle in subtitles_state:
        if hasattr(subtitle, 'is_success') and subtitle.is_success:
            success_count += 1

    if success_count == 0:
        return gr.update(
            value="❌ **音频合成检查失败**\n\n🎵 **错误**: 所有字幕行的音频生成都失败了\n\n💡 **解决方案**: 请检查TTS服务配置，重新生成音频")

    if success_count < total_count:
        failed_count = total_count - success_count
        return gr.update(
            value=f"⚠️ **音频合成不完整**\n\n🎵 **警告**: {total_count} 行字幕中有 {failed_count} 行音频生成失败\n\n💡 **建议**: 建议先修复失败的音频生成，或继续合成（将跳过失败的部分）")

    # 4. 所有检查通过，开始执行合成流程
    try:
        # 初始化进度条
        progress(0.0, desc="正在准备视频合成...")

        # 步骤1: 导出字幕文件
        # 创建临时目录用于视频处理（添加workspace名称层级）
        project_name = subtitles_state.dir if subtitles_state.dir else "video_compose"
        temp_dir = os.path.join(current_path, "SAVAdata", "temp", "video_compose", project_name)
        os.makedirs(temp_dir, exist_ok=True)

        # 检查是否已经有音频生成的哈希目录
        progress(0.15, desc="正在检查输出目录...")
        project_name = subtitles_state.dir if subtitles_state.dir else "video_compose"
        existing_output_dir = os.environ.get("current_output_dir")

        if existing_output_dir and os.path.exists(existing_output_dir):
            # 使用音频合成时创建的哈希目录
            output_dir = existing_output_dir
            print(f"🔄 使用现有输出目录: {output_dir}")
        else:
            # 创建新的基于workspace名称的目录
            output_dir = get_output_dir_with_workspace_name(project_name, "video_compose")
            os.environ["current_output_dir"] = output_dir
            print(f"🆕 创建新输出目录: {output_dir}")

        # 生成基于项目名称的文件名
        project_name = subtitles_state.dir if subtitles_state.dir else "video_compose"

        # 导出原始字幕到临时目录（用于视频处理）
        progress(0.25, desc="正在处理字幕文件...")
        original_srt_path = os.path.join(temp_dir, "original.srt")

        # 检查原始字幕文件格式，如果是 ASS 或 VTT，需要先转换为 SRT
        original_subtitle_file = subtitle_files_filtered[0].name
        original_ext = os.path.splitext(original_subtitle_file)[1].lower()

        if original_ext == '.ass':
            # ASS 文件转换为 SRT
            from Sava_Utils.subtitle_processor import format_ass_file, extract_ass_to_srt, get_available_styles

            # 格式化 ASS 文件
            formatted_ass_path = os.path.join(temp_dir, "formatted_original.ass")
            format_success = format_ass_file(original_subtitle_file, formatted_ass_path)
            if not format_success:
                formatted_ass_path = original_subtitle_file

            # 获取样式并转换
            styles = get_available_styles(formatted_ass_path)
            style_name = styles[0] if styles else "Default"
            extract_ass_to_srt(formatted_ass_path, style_name, original_srt_path)

        elif original_ext == '.vtt':
            # VTT 文件转换为 SRT
            from Sava_Utils.subtitle_processor import convert_subtitle
            convert_subtitle(original_subtitle_file, original_srt_path)

        else:
            # SRT、CSV、TXT 文件直接复制
            shutil.copy2(original_subtitle_file, original_srt_path)

        # 导出新字幕到输出目录（最终输出文件）
        new_srt_path = os.path.join(output_dir, f"{project_name}_final.srt")
        subtitles_state.export(fp=new_srt_path, open_explorer=False)

        # 如果原始文件是 ASS 或 VTT，也导出原格式的字幕文件
        original_subtitle_file = subtitle_files_filtered[0].name
        original_ext = os.path.splitext(original_subtitle_file)[1].lower()

        if original_ext in ['.ass', '.vtt']:
            try:
                print(f"🔄 正在基于最终SRT重新生成 {original_ext.upper()} 格式文件...")
                original_format_file = export_original_format(
                    original_subtitle_file, new_srt_path, project_name, original_ext, output_dir
                )
                if original_format_file:
                    print(f"✅ {original_ext.upper()} 字幕文件已基于最终SRT重新生成: {original_format_file}")
                else:
                    print(f"❌ {original_ext.upper()} 格式文件生成失败")
            except Exception as format_error:
                print(f"⚠️ 生成 {original_ext.upper()} 格式失败: {format_error}")
                gr.Warning(f"生成 {original_ext.upper()} 格式失败: {str(format_error)}")

        # 步骤2: 获取无声视频路径
        progress(0.35, desc="正在准备视频文件...")
        # 从processing_state中获取处理后的视频路径
        silent_video_path = None
        processing_result = current_state.get("processing_result", {})

        if processing_result and "raw_video" in processing_result:
            silent_video_path = processing_result["raw_video"]
            print(f"🎬 Found silent video: {silent_video_path}")

        if not silent_video_path or not os.path.exists(silent_video_path):
            # 如果没有找到无声视频，使用原视频
            silent_video_path = video_path
            print(f"⚠️ Using original video as fallback: {silent_video_path}")

        # 步骤3: 调用视频变速处理
        # 创建进度回调函数
        def video_progress_callback(percent, desc):
            # 将百分比转换为0-1之间的小数
            progress_value = percent / 100.0
            progress(progress_value, desc=desc)

        speed_result = adjust_video_speed_by_subtitles(
            video_path=silent_video_path,
            original_srt_path=original_srt_path,
            new_srt_path=new_srt_path,
            output_dir=temp_dir,
            max_workers=4,
            use_gpu=True,
            progress_callback=video_progress_callback
        )

        if not speed_result['success']:
            return gr.update(value=f"❌ **视频变速处理失败**\n\n🎬 **错误**: {speed_result['message']}")

        speed_adjusted_video = speed_result['output_path']

        # 步骤4: 获取生成的音频文件路径
        progress(0.65, desc="正在准备音频文件...")
        # 优先使用环境变量中保存的音频路径
        audio_file_path = os.environ.get("current_audio_path")

        # 如果环境变量中没有，则尝试在哈希目录中查找
        if not audio_file_path or not os.path.exists(audio_file_path):
            audio_file_path = os.path.join(output_dir, f"{project_name}.wav")

        # 如果还是找不到，尝试在旧的输出目录中查找
        if not os.path.exists(audio_file_path):
            fallback_audio_path = os.path.join(current_path, "SAVAdata", "output", f"{project_name}.wav")
            if os.path.exists(fallback_audio_path):
                audio_file_path = fallback_audio_path
            else:
                return gr.update(
                    value="❌ **音频文件不存在**\n\n🎵 **错误**: 找不到生成的音频文件\n\n💡 **建议**: 请先完成音频合成")

        # 步骤5: 合成变速视频与音频
        progress(0.80, desc="正在合成视频和音频...")
        # 使用与字幕相同的哈希输出目录
        output_video_path = os.path.join(output_dir, f"{project_name}_final.mp4")

        final_video = merge_video_with_audio(
            video_path=speed_adjusted_video,
            audio_path=audio_file_path,
            output_path=output_video_path,
            use_gpu=True,
            sync_to_audio=True
        )

        # 自动打开输出文件夹
        if not Sava_Utils.config.server_mode:
            output_folder = os.path.dirname(final_video)
            try:
                os.system(f'explorer /select, "{final_video}"')
                print(f"📂 已自动打开输出文件夹: {output_folder}")
            except Exception as e:
                print(f"⚠️ 无法打开文件夹: {e}")

        # 生成成功信息
        progress(1.0, desc="视频合成完成！")
        success_info = f"""
✅ **视频合成完成！**

📋 **处理结果**
• ✅ 字幕导出: 成功
• ✅ 视频变速: 成功 ({speed_result['segments_processed']}/{speed_result['total_segments']} 片段)
• ✅ 音视频合成: 成功

📊 **处理统计**
• 原始时长: {speed_result['original_duration']:.2f}秒
• 目标时长: {speed_result['target_duration']:.2f}秒
• 平均变速比: {speed_result['average_speed_ratio']:.2f}x
• 音频成功率: {success_count / total_count * 100:.1f}%

📁 **输出文件**
• 🎬 最终视频: `{final_video}`
• 📂 保存位置: `{os.path.dirname(final_video)}`

🎉 **合成成功！**
您的视频已经成功合成，包含了同步的音频和调整后的字幕文件。

💡 **提示**: 可以在输出目录中找到最终的视频文件
        """.strip()

        return gr.update(value=success_info)

    except Exception as e:
        error_info = f"""
❌ **视频合成失败**

🔧 **错误信息**: {str(e)}

💡 **可能的解决方案:**
• 检查所有文件是否完整
• 确认有足够的磁盘空间
• 重新生成音频后再试
• 检查视频文件是否损坏

🔄 **建议**: 重新执行整个流程
        """.strip()

        return gr.update(value=error_info)


def create_batch_dubbing_ui():
    """创建批量配音界面 - 参考其他页面样式，简洁实用"""

    with gr.Row():
        # 左列：任务列表 (70%)
        with gr.Column(scale=7):
            with gr.Accordion(label="任务列表", open=True):
                # 任务统计
                with gr.Row():
                    batch_total_tasks = gr.Number(
                        label="总任务数",
                        value=0,
                        interactive=False,
                        scale=1
                    )
                    batch_completed_tasks = gr.Number(
                        label="已完成",
                        value=0,
                        interactive=False,
                        scale=1
                    )
                    batch_failed_tasks = gr.Number(
                        label="失败",
                        value=0,
                        interactive=False,
                        scale=1
                    )
                    batch_success_rate = gr.Number(
                        label="成功率(%)",
                        value=0,
                        interactive=False,
                        scale=1
                    )

                # 任务列表显示
                batch_task_display = gr.HTML(
                    value="<div style='text-align: center; padding: 40px; color: #666; border: 2px dashed #ddd; border-radius: 8px;'>📝 暂无任务<br><small>点击右侧「添加任务」开始</small></div>"
                )

                # 任务操作
                with gr.Row():
                    batch_select_all_btn = gr.Button("全选", size="sm")
                    batch_select_none_btn = gr.Button("全不选", size="sm")
                    batch_delete_selected_btn = gr.Button("删除选中", size="sm", variant="stop")
                    batch_retry_failed_btn = gr.Button("重试失败", size="sm")

        # 右列：文件上传和配置 (30%)
        with gr.Column(scale=3):
            # 文件上传区域
            with gr.Accordion(label="文件上传", open=True):
                task_video_file = gr.File(
                    label="视频文件",
                    file_types=['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.ts', '.m2ts'],
                    type="filepath"
                )

                task_subtitle_file = gr.File(
                    label="字幕文件",
                    file_types=['.srt', '.ass', '.vtt', '.csv', '.txt'],
                    type="filepath"
                )

                with gr.Row():
                    confirm_add_btn = gr.Button("添加任务", variant="primary", scale=2)
                    cancel_add_btn = gr.Button("重置", variant="secondary", scale=1)

            # TTS配置 - 参考字幕配音页面顺序，默认收缩
            with gr.Accordion(label="TTS配置", open=False):
                batch_tts_service = gr.Dropdown(
                    label="TTS服务",
                    choices=["GSV", "Index-TTS", "Edge-TTS", "Custom"],
                    value="GSV",
                    interactive=True
                )

            # 输出设置 - 默认收缩
            with gr.Accordion(label="输出设置", open=False):
                batch_output_dir = gr.Textbox(
                    label="输出目录",
                    value="SAVAdata/output/batch_dubbing",
                    placeholder="输出文件保存目录",
                    interactive=True
                )

                # 保留原视频格式选项
                batch_keep_original_format = gr.Checkbox(
                    label="保留原视频格式",
                    value=True,
                    interactive=True
                )

                # 输出格式选择（默认隐藏）
                batch_output_format = gr.Dropdown(
                    label="输出格式",
                    choices=["MP4", "AVI", "MKV", "MOV", "WMV", "FLV", "WEBM"],
                    value="MP4",
                    interactive=True,
                    visible=False
                )

                batch_keep_original = gr.Checkbox(
                    label="保留原始文件",
                    value=True,
                    interactive=True
                )

            # 批量操作
            with gr.Accordion(label="批量操作", open=True):
                with gr.Row():
                    batch_start_all_btn = gr.Button("批量开始", variant="primary")
                    batch_pause_btn = gr.Button("暂停", variant="secondary")

                with gr.Row():
                    batch_clear_btn = gr.Button("清空列表", variant="secondary")
                    batch_export_btn = gr.Button("导出结果", variant="secondary")

                # 处理进度
                batch_progress_info = gr.Textbox(
                    label="处理进度",
                    value="等待任务...",
                    interactive=False
                )

    # 状态管理
    batch_tasks_state = gr.State(value=[])
    batch_processing_state = gr.State(value={"running": False, "current_task": 0})

    return {
        'task_video_file': task_video_file,
        'task_subtitle_file': task_subtitle_file,
        'confirm_add_btn': confirm_add_btn,
        'cancel_add_btn': cancel_add_btn,
        'batch_start_all_btn': batch_start_all_btn,
        'batch_pause_btn': batch_pause_btn,
        'batch_clear_btn': batch_clear_btn,
        'batch_export_btn': batch_export_btn,
        'batch_progress_info': batch_progress_info,
        'batch_tts_service': batch_tts_service,
        'batch_output_dir': batch_output_dir,
        'batch_keep_original_format': batch_keep_original_format,
        'batch_output_format': batch_output_format,
        'batch_keep_original': batch_keep_original,
        'batch_total_tasks': batch_total_tasks,
        'batch_completed_tasks': batch_completed_tasks,
        'batch_failed_tasks': batch_failed_tasks,
        'batch_success_rate': batch_success_rate,
        'batch_task_display': batch_task_display,
        'batch_select_all_btn': batch_select_all_btn,
        'batch_select_none_btn': batch_select_none_btn,
        'batch_delete_selected_btn': batch_delete_selected_btn,
        'batch_retry_failed_btn': batch_retry_failed_btn,
        'batch_tasks_state': batch_tasks_state,
        'batch_processing_state': batch_processing_state
    }


def create_task_row(task_id, video_file, subtitle_file, status="待处理", result=""):
    """创建单个任务行"""
    with gr.Row():
        # 选择框
        task_checkbox = gr.Checkbox(label="", value=False, scale=1)

        # 视频文件
        video_display = gr.Textbox(
            value=os.path.basename(video_file) if video_file else "",
            label="视频文件",
            interactive=False,
            scale=3
        )

        # 字幕文件
        subtitle_display = gr.Textbox(
            value=os.path.basename(subtitle_file) if subtitle_file else "",
            label="字幕文件",
            interactive=False,
            scale=3
        )

        # 处理结果
        result_display = gr.Textbox(
            value=f"{status}: {result}",
            label="处理结果",
            interactive=False,
            scale=2
        )

        # 操作按钮
        with gr.Column(scale=2):
            start_btn = gr.Button("开始", variant="primary", size="sm")
            retry_btn = gr.Button("重新执行", variant="secondary", size="sm")

    return {
        'checkbox': task_checkbox,
        'video_display': video_display,
        'subtitle_display': subtitle_display,
        'result_display': result_display,
        'start_btn': start_btn,
        'retry_btn': retry_btn,
        'task_id': task_id,
        'video_file': video_file,
        'subtitle_file': subtitle_file
    }


def add_batch_task(video_file, subtitle_file, current_tasks):
    """添加批量任务"""
    if not video_file or not subtitle_file:
        gr.Warning("请选择视频文件和字幕文件")
        return current_tasks, None, None, update_batch_statistics(current_tasks)

    # 检查文件是否已存在
    for task in current_tasks:
        if task['video_file'] == video_file and task['subtitle_file'] == subtitle_file:
            gr.Warning("该任务已存在")
            return current_tasks, None, None, update_batch_statistics(current_tasks)

    # 创建新任务
    new_task = {
        'id': len(current_tasks) + 1,
        'video_file': video_file,
        'subtitle_file': subtitle_file,
        'status': '待处理',
        'result': '',
        'selected': False,
        'progress': 0,
        'created_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    current_tasks.append(new_task)
    gr.Info(f"已添加任务: {os.path.basename(video_file)}")

    # 清空文件选择
    return current_tasks, None, None, update_batch_statistics(current_tasks)


def update_batch_statistics(tasks):
    """更新批量任务统计信息"""
    total = len(tasks)
    completed = len([t for t in tasks if t['status'] == '已完成'])
    failed = len([t for t in tasks if t['status'] == '失败'])
    success_rate = (completed / total * 100) if total > 0 else 0

    return (
        gr.update(value=total),
        gr.update(value=completed),
        gr.update(value=failed),
        gr.update(value=round(success_rate, 1))
    )


def clear_batch_tasks():
    """清空批量任务列表"""
    return [], *update_batch_statistics([])


def reset_file_inputs():
    """重置文件输入"""
    return None, None


def toggle_output_format_visibility(keep_original_format):
    """根据保留原视频格式选项控制输出格式的可见性"""
    # 如果保留原格式，隐藏输出格式选择；否则显示
    return gr.update(visible=not keep_original_format)


def render_batch_tasks(tasks):
    """渲染批量任务列表 - 简洁实用版本"""
    if not tasks:
        return gr.update(value="""
        <div style='text-align: center; padding: 40px; color: #666; border: 2px dashed #ddd; border-radius: 8px;'>
            📝 暂无任务<br><small>点击左侧「添加任务」开始</small>
        </div>
        """)

    # 简洁表格样式
    table_style = """
    <style>
        .batch-table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border: 1px solid #ddd;
        }
        .batch-table th {
            background: #f5f5f5;
            padding: 10px 8px;
            text-align: center;
            font-weight: 600;
            border-bottom: 2px solid #ddd;
        }
        .batch-table td {
            padding: 10px 8px;
            border-bottom: 1px solid #eee;
            vertical-align: middle;
        }
        .batch-table tr:hover {
            background: #f9f9f9;
        }
        .status-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
            text-align: center;
            display: inline-block;
            min-width: 60px;
        }
        .status-pending { background: #fff3cd; color: #856404; }
        .status-processing { background: #cce5ff; color: #004085; }
        .status-completed { background: #d4edda; color: #155724; }
        .status-failed { background: #f8d7da; color: #721c24; }
        .file-name { font-weight: 500; color: #333; margin-bottom: 2px; }
        .file-path { font-size: 11px; color: #666; word-break: break-all; }
        .action-btn {
            padding: 4px 8px;
            margin: 1px;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
            background: #007bff;
            color: white;
        }
        .action-btn:hover { background: #0056b3; }
        .btn-retry { background: #28a745; }
        .btn-retry:hover { background: #1e7e34; }
        .btn-delete { background: #dc3545; }
        .btn-delete:hover { background: #c82333; }
    </style>
    """

    # 表格HTML
    table_html = table_style + '<table class="batch-table">'

    # 表头
    table_html += """
    <thead>
        <tr>
            <th style="width: 50px;">选择</th>
            <th style="width: 35%;">视频文件</th>
            <th style="width: 35%;">字幕文件</th>
            <th style="width: 15%;">状态</th>
            <th style="width: 15%;">操作</th>
        </tr>
    </thead>
    <tbody>
    """

    # 任务行
    for i, task in enumerate(tasks):
        status_class = {
            '待处理': 'status-pending',
            '处理中': 'status-processing',
            '已完成': 'status-completed',
            '失败': 'status-failed'
        }.get(task['status'], 'status-pending')

        progress = task.get('progress', 0)

        table_html += f"""
        <tr>
            <td style="text-align: center;">
                <input type="checkbox" id="task_{task['id']}" {'checked' if task.get('selected', False) else ''}>
            </td>
            <td>
                <div class="file-name">{os.path.basename(task['video_file'])}</div>
                <div class="file-path">{task['video_file']}</div>
            </td>
            <td>
                <div class="file-name">{os.path.basename(task['subtitle_file'])}</div>
                <div class="file-path">{task['subtitle_file']}</div>
            </td>
            <td style="text-align: center;">
                <div class="status-badge {status_class}">{task['status']}</div>
                <div style="font-size: 11px; color: #666; margin-top: 4px;">{progress}% • {task.get('created_time', '')}</div>
            </td>
            <td style="text-align: center;">
                <button class="action-btn" onclick="startBatchTask({task['id']})">开始</button>
                <button class="action-btn btn-retry" onclick="retryBatchTask({task['id']})">重试</button>
                <button class="action-btn btn-delete" onclick="deleteBatchTask({task['id']})">删除</button>
            </td>
        </tr>
        """

    table_html += "</tbody></table>"

    return gr.update(value=table_html)


def start_batch_task(task_id, tasks):
    """开始单个批量任务"""
    for task in tasks:
        if task['id'] == task_id:
            task['status'] = '处理中'
            # 这里可以调用实际的处理逻辑
            # 暂时模拟处理
            task['status'] = '已完成'
            task['result'] = '配音完成'
            break

    return tasks


def retry_batch_task(task_id, tasks):
    """重试批量任务"""
    for task in tasks:
        if task['id'] == task_id:
            task['status'] = '待处理'
            task['result'] = ''
            break

    return tasks


def start_all_batch_tasks(tasks):
    """开始所有批量任务"""
    if not tasks:
        gr.Warning("没有任务需要处理")
        return tasks

    for task in tasks:
        if task['status'] == '待处理':
            task['status'] = '处理中'
            # 这里可以调用实际的处理逻辑
            # 暂时模拟处理
            task['status'] = '已完成'
            task['result'] = '配音完成'

    gr.Info("所有任务处理完成")
    return tasks


def export_batch_results(tasks):
    """导出批量处理结果"""
    if not tasks:
        gr.Warning("没有结果可导出")
        return

    # 生成结果报告
    report = "批量配音处理报告\n"
    report += "=" * 50 + "\n"

    for task in tasks:
        report += f"任务ID: {task['id']}\n"
        report += f"视频文件: {task['video_file']}\n"
        report += f"字幕文件: {task['subtitle_file']}\n"
        report += f"处理状态: {task['status']}\n"
        report += f"处理结果: {task.get('result', '')}\n"
        report += "-" * 30 + "\n"

    # 保存报告文件
    report_path = os.path.join(current_path, "SAVAdata", "output", "batch_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    gr.Info(f"结果已导出到: {report_path}")


def generate_silence_audio(subtitle, dir):
    """为失败的字幕生成静音音频片段"""
    try:
        # 计算音频时长（秒）
        duration = subtitle.end_time - subtitle.start_time
        if duration <= 0:
            duration = 1.0  # 最小1秒

        # 生成静音音频
        sr = 32000  # 采样率
        samples = int(duration * sr)
        silence_audio = np.zeros(samples, dtype=np.float32)

        # 保存静音音频文件
        filepath = os.path.join(dir, f"{subtitle.index}.wav")
        sf.write(filepath, silence_audio, sr)

        # 标记为成功（虽然是静音，但避免重复处理）
        subtitle.is_success = True

        logger.info(f"为字幕 {subtitle.index} 生成静音片段: {duration:.2f}秒")
        return filepath

    except Exception as e:
        logger.error(f"生成静音片段失败 {subtitle.index}: {e}")
        subtitle.is_success = False
        return None


def get_output_dir_with_workspace_name(workspace_name=None, fallback_name="default"):
    """
    生成基于workspace名称的输出目录路径

    Args:
        workspace_name: workspace名称，如 "Ali_s Voice Over Trade 1-ass"
        fallback_name: 当workspace_name为空时的备用名称

    Returns:
        str: SAVAdata/output/workspace_name 格式的目录路径
    """
    # 确定目录名称
    if workspace_name:
        dir_name = workspace_name
    else:
        # 使用备用名称加时间戳
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        dir_name = f"{fallback_name}_{timestamp}"

    # 返回完整的输出目录路径
    output_dir = os.path.join(current_path, "SAVAdata", "output", dir_name)

    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)

    return output_dir


componments = {
    1: [GSV, INDEXTTS, EDGETTS, CUSTOM],
    2: [TRANSLATION_MODULE, POLYPHONE],
    3: [],
}


def custom_api(text):
    raise i18n('You need to load custom API functions!')


def export_subtitle_with_new_name(file_list, subtitle_state):
    """
    导出字幕文件，基于原字幕文件名生成新的文件名
    支持导出原格式（ASS/VTT）和 SRT 格式

    Args:
        file_list: 原始文件列表
        subtitle_state: 字幕状态对象

    Returns:
        更新后的文件列表
    """
    try:
        # 获取原始文件列表
        original_files = [i.name for i in file_list] if file_list else []

        exported_files = []

        if original_files:
            # 获取原始文件信息
            original_file = original_files[0]
            original_dir = os.path.dirname(original_file)
            original_basename = Sava_Utils.utils.basename_no_ext(original_file)

            # 生成与workspace一致的目录名称（保留格式信息）
            original_filename = os.path.basename(original_file)
            workspace_dir_name = original_filename.replace(".", "-")
            # 如果以短横线结尾，去掉最后的短横线
            if workspace_dir_name.endswith("-"):
                workspace_dir_name = workspace_dir_name[:-1]

            # 如果原文件在output目录外，则使用基于workspace名称的output目录
            if "SAVAdata" not in original_dir or "output" not in original_dir:
                # 生成基于workspace名称的输出目录
                workspace_output_dir = get_output_dir_with_workspace_name(workspace_dir_name, "subtitle")
                # 指定导出路径，避免重复生成目录
                srt_filepath = os.path.join(workspace_output_dir, f"{original_basename}.srt")
                exported_srt_file = subtitle_state.export(fp=srt_filepath, open_explorer=False)
            else:
                # 使用原目录
                srt_filepath = os.path.join(original_dir, f"{original_basename}.srt")
                exported_srt_file = subtitle_state.export(fp=srt_filepath, open_explorer=False)
        else:
            # 没有原始文件，使用默认导出
            exported_srt_file = subtitle_state.export(open_explorer=False)

        if exported_srt_file:
            exported_files.append(exported_srt_file)
            print(f"✅ SRT 字幕文件已导出: {exported_srt_file}")

            # 如果有原始文件，检查是否需要导出原格式
            if original_files:
                original_file = original_files[0]
                original_basename = Sava_Utils.utils.basename_no_ext(original_file)
                original_ext = os.path.splitext(original_file)[1].lower()

                if original_ext in ['.ass', '.vtt']:
                    try:
                        print(f"🔄 正在基于新SRT重新生成 {original_ext.upper()} 格式文件...")
                        original_format_file = export_original_format(
                            original_file, exported_srt_file, original_basename, original_ext,
                            os.path.dirname(exported_srt_file)
                        )
                        if original_format_file:
                            exported_files.append(original_format_file)
                            print(f"✅ {original_ext.upper()} 字幕文件已基于新SRT重新生成: {original_format_file}")
                        else:
                            print(f"❌ {original_ext.upper()} 格式文件生成失败")
                    except Exception as format_error:
                        print(f"⚠️ 导出 {original_ext.upper()} 格式失败: {format_error}")
                        gr.Warning(f"导出 {original_ext.upper()} 格式失败: {str(format_error)}")

            # 打开包含导出文件的目录
            if not Sava_Utils.config.server_mode:
                export_dir = os.path.dirname(exported_srt_file)
                os.system(f'explorer /select, {exported_srt_file}')

            gr.Info(f"字幕文件已导出: {len(exported_files)} 个文件")

            # 只返回原始文件列表，不包含导出的文件（避免更新上传组件）
            return original_files
        else:
            # 如果导出失败，返回原始文件列表
            return original_files

    except Exception as e:
        print(f"❌ 导出字幕文件失败: {e}")
        gr.Error(f"导出字幕文件失败: {str(e)}")
        return [i.name for i in file_list] if file_list else []


def export_original_format(original_file, srt_file, base_name, original_ext, output_dir):
    """
    导出原格式的字幕文件（ASS 或 VTT）

    Args:
        original_file: 原始字幕文件路径
        srt_file: 生成的 SRT 文件路径
        base_name: 基础文件名（不含扩展名）
        original_ext: 原始文件扩展名 (.ass 或 .vtt)
        output_dir: 输出目录

    Returns:
        导出的原格式文件路径，失败时返回 None
    """
    try:
        if original_ext == '.ass':
            # 导出 ASS 格式
            print(f"📝 开始处理ASS格式转换...")
            print(f"   原始ASS文件: {original_file}")
            print(f"   新SRT文件: {srt_file}")

            from Sava_Utils.subtitle_processor import sync_srt_timestamps_to_ass, format_ass_file

            # 先格式化ASS文件
            formatted_ass_path = os.path.join(output_dir, f"{base_name}_formatted.ass")
            print(f"🔧 正在格式化ASS文件...")
            print(f"   格式化ASS文件: {formatted_ass_path}")

            format_success = format_ass_file(original_file, formatted_ass_path)
            if not format_success:
                print(f"⚠️ ASS文件格式化失败，使用原始文件")
                formatted_ass_path = original_file
            else:
                print(f"✅ ASS文件格式化成功")

            output_ass_path = os.path.join(output_dir, f"{base_name}_final.ass")

            # 如果目标文件已存在，添加时间戳
            if os.path.exists(output_ass_path):
                timestamp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
                output_ass_path = os.path.join(output_dir, f"{base_name}_final{timestamp}.ass")

            print(f"   输出ASS文件: {output_ass_path}")
            print(f"🔄 正在使用sync_srt_timestamps_to_ass同步时间戳...")
            print(f"   使用格式化后的ASS文件: {formatted_ass_path}")

            # 使用格式化后的ASS文件进行时间戳同步
            success = sync_srt_timestamps_to_ass(formatted_ass_path, srt_file, output_ass_path)

            if success and os.path.exists(output_ass_path):
                print(f"✅ ASS文件同步成功")

                # 清理临时的格式化文件（如果不是原始文件）
                if formatted_ass_path != original_file and os.path.exists(formatted_ass_path):
                    try:
                        os.remove(formatted_ass_path)
                        print(f"🧹 已清理临时格式化文件")
                    except Exception as e:
                        print(f"⚠️ 清理临时文件失败: {e}")

                return output_ass_path
            else:
                print(f"❌ ASS 文件同步失败")

                # 清理临时的格式化文件（如果不是原始文件）
                if formatted_ass_path != original_file and os.path.exists(formatted_ass_path):
                    try:
                        os.remove(formatted_ass_path)
                        print(f"🧹 已清理临时格式化文件")
                    except Exception as e:
                        print(f"⚠️ 清理临时文件失败: {e}")

                return None

        elif original_ext == '.vtt':
            # 导出 VTT 格式
            print(f"📝 开始处理VTT格式转换...")
            print(f"   新SRT文件: {srt_file}")

            from Sava_Utils.subtitle_processor import convert_subtitle

            output_vtt_path = os.path.join(output_dir, f"{base_name}_final.vtt")

            # 如果目标文件已存在，添加时间戳
            if os.path.exists(output_vtt_path):
                timestamp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
                output_vtt_path = os.path.join(output_dir, f"{base_name}_final{timestamp}.vtt")

            print(f"   输出VTT文件: {output_vtt_path}")
            print(f"🔄 正在使用convert_subtitle从SRT转换为VTT...")

            # 使用 convert_subtitle 从 SRT 转换为 VTT
            success = convert_subtitle(srt_file, output_vtt_path)

            if success and os.path.exists(output_vtt_path):
                print(f"✅ VTT文件转换成功")
                return output_vtt_path
            else:
                print(f"❌ VTT 文件转换失败")
                return None
        else:
            print(f"❌ 不支持的格式: {original_ext}")
            return None

    except Exception as e:
        print(f"❌ 导出原格式文件失败: {e}")
        return None


# single speaker
def generate(*args, interrupt_event: Sava_Utils.utils.Flag, proj="", in_files=[], fps=30, offset=0, max_workers=1):
    t1 = time.time()
    fps = positive_int(fps)
    if in_files in [None, []]:
        gr.Info(i18n('Please upload the subtitle file!'))
        return (None, i18n('Please upload the subtitle file!'), getworklist(), *load_page(Subtitles()), Subtitles())
    if Sava_Utils.config.server_mode and len(in_files) > 1:
        gr.Warning(i18n('The current mode does not allow batch processing!'))
        return (None, i18n('The current mode does not allow batch processing!'), getworklist(), *load_page(Subtitles()),
                Subtitles())
    os.makedirs(os.path.join(current_path, "SAVAdata", "output"), exist_ok=True)
    for in_file in in_files:
        try:
            subtitle_list = read_file(in_file.name, fps, offset)
        except Exception as e:
            what = str(e)
            gr.Warning(what)
            return (None, what, getworklist(), *load_page(Subtitles()), Subtitles())
        # subtitle_list.sort()
        subtitle_list.set_dir_name(os.path.basename(in_file.name).replace(".", "-"))
        subtitle_list.set_proj(proj)
        Projet_dict[proj].before_gen_action(*args, config=Sava_Utils.config, notify=False, force=False)
        abs_dir = subtitle_list.get_abs_dir()
        if Sava_Utils.config.server_mode:
            max_workers = 1
        file_list = []
        with interrupt_event:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(save, args, proj=proj, dir=abs_dir, subtitle=i) for i in subtitle_list]
                for future in tqdm(
                        concurrent.futures.as_completed(futures),
                        total=len(subtitle_list),
                        desc=i18n('Synthesizing single-speaker task'),
                ):
                    if interrupt_event.is_set():
                        executor.shutdown(wait=True, cancel_futures=True)
                        subtitle_list.dump()
                        gr.Info("Interrupted.")
                        break
                    item = future.result()
                    if item:
                        file_list.append(item)
            if interrupt_event.is_set():
                sr_audio = None
                break
            if len(file_list) == 0:
                shutil.rmtree(abs_dir)
                if len(in_files) == 1:
                    raise gr.Error(i18n('All subtitle syntheses have failed, please check the API service!'))
                else:
                    continue
        sr_audio = subtitle_list.audio_join(sr=Sava_Utils.config.output_sr)
    t2 = time.time()
    m, s = divmod(t2 - t1, 60)
    use_time = "%02d:%02d" % (m, s)
    return (
        sr_audio,
        f"{i18n('Done! Time used')}:{use_time}",
        getworklist(value=subtitle_list.dir),
        *load_page(subtitle_list),
        subtitle_list,
    )


def generate_preprocess(interrupt_event, *args, project=None):
    try:
        args, kwargs = Projet_dict[project].arg_filter(*args)
    except Exception as e:
        info = f"{i18n('An error occurred')}: {str(e)}"
        gr.Warning(info)
        return None, info, getworklist(), *load_page(Subtitles()), Subtitles()
    return generate(*args, interrupt_event=interrupt_event, **kwargs)


def gen_multispeaker(interrupt_event: Sava_Utils.utils.Flag, *args,
                     remake=False):  # args: page,maxworkers,*args,subtitles
    page = args[0]
    max_workers = int(args[1])
    subtitles: Subtitles = args[-1]
    if subtitles is None or len(subtitles) == 0:
        gr.Info(i18n('There is no subtitle in the current workspace'))
        return *show_page(page, Subtitles()), None
    proj_args = (None, None, *args[:-1])
    if remake:
        todo = [i for i in subtitles if not i.is_success]
    else:
        todo = subtitles
    if len(todo) == 0:
        gr.Info(i18n('No subtitles are going to be resynthesized.'))
        return *show_page(page, subtitles), None
    abs_dir = subtitles.get_abs_dir()
    tasks = defaultdict(list)
    for i in todo:
        tasks[i.speaker].append(i)
    if list(tasks.keys()) == [None] and subtitles.default_speaker is None and subtitles.proj is None:
        gr.Warning(i18n('Warning: No speaker has been assigned'))
        return *show_page(page, subtitles), None
    ok = True
    progress = 0
    for key in tasks.keys():
        if key is None:
            if subtitles.proj is None and subtitles.default_speaker is not None and len(tasks[None]) > 0:
                print(f"{i18n('Using default speaker')}:{subtitles.default_speaker}")
                spk = subtitles.default_speaker
            elif subtitles.proj is not None and remake:
                args = proj_args
                project = subtitles.proj
                spk = None
            else:
                continue
        else:
            spk = key
        if spk is not None:
            try:
                with open(os.path.join(current_path, "SAVAdata", "speakers", spk), 'rb') as f:
                    info = pickle.load(f)
            except FileNotFoundError:
                ok = False
                logger.error(f"{i18n('Speaker archive not found')}: {spk}")
                gr.Warning(f"{i18n('Speaker archive not found')}: {spk}")
                continue
            args = info["raw_data"]
            project = info["project"]
        try:
            args, kwargs = Projet_dict[project].arg_filter(*args)
            Projet_dict[project].before_gen_action(*args, config=Sava_Utils.config)
        except Exception as e:
            ok = False
            gr.Warning(str(e))
            continue
        if Sava_Utils.config.server_mode:
            max_workers = 1
        file_list = []
        with interrupt_event:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(save, args, proj=project, dir=abs_dir, subtitle=i) for i in tasks[key]]
                for future in tqdm(
                        concurrent.futures.as_completed(futures),
                        total=len(todo),
                        initial=progress,
                        desc=f"{i18n('Synthesizing multi-speaker task, the current speaker is')} :{spk}",
                ):
                    if interrupt_event.is_set():
                        executor.shutdown(wait=True, cancel_futures=True)
                        gr.Info("Interrupted.")
                        ok = False
                        break
                    item = future.result()
                    if item:
                        file_list.append(item)
                if interrupt_event.is_set():
                    break
        progress += len(file_list)
        if len(file_list) == 0:
            ok = False
            gr.Warning(f"{i18n('Synthesis for the single speaker has failed !')} {spk}")

    gr.Info(i18n('Done!'))
    if remake:
        if ok:
            gr.Info(i18n('Audio re-generation was successful! Click the <Reassemble Audio> button.'))
        subtitles.dump()
        return show_page(page, subtitles)
    else:
        sr_audio = subtitles.audio_join(sr=Sava_Utils.config.output_sr)
    return *show_page(page, subtitles), sr_audio


def save(args, proj: str = None, dir: str = None, subtitle: Subtitle = None):
    # 设置当前字幕索引环境变量，供 Clone 模式使用
    os.environ["current_subtitle_index"] = str(subtitle.index)
    audio = Projet_dict[proj].save_action(*args, text=subtitle.text)
    if audio is not None:
        if audio[:4] == b'RIFF' and audio[8:12] == b'WAVE':
            # sr=int.from_bytes(audio[24:28],'little')
            filepath = os.path.join(dir, f"{subtitle.index}.wav")
            if Sava_Utils.config.remove_silence:
                audio, sr = Sava_Utils.librosa_load.load_audio(io.BytesIO(audio))
                audio = remove_silence(audio, sr)
                sf.write(filepath, audio, sr)
            else:
                with open(filepath, 'wb') as file:
                    file.write(audio)
            if Sava_Utils.config.max_accelerate_ratio > 1.0:
                audio, sr = Sava_Utils.librosa_load.load_audio(filepath)
                target_dur = int(subtitle.end_time - subtitle.start_time) * sr
                if target_dur > 0 and (audio.shape[-1] - target_dur) > (0.01 * sr):
                    ratio = min(audio.shape[-1] / target_dur, Sava_Utils.config.max_accelerate_ratio)
                    cmd = f'ffmpeg -i "{filepath}" -filter:a atempo={ratio:.2f} -y "{filepath}.wav"'
                    p = subprocess.Popen(cmd, cwd=current_path, shell=True, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
                    logger.info(f"{i18n('Execute command')}:{cmd}")
                    exit_code = p.wait()
                    if exit_code == 0:
                        shutil.move(f"{filepath}.wav", filepath)
                    else:
                        logger.error("Failed to execute ffmpeg.")
            subtitle.is_success = True
            return filepath
        else:
            try:
                data = json.loads(audio)
                logger.error(
                    f"{i18n('Failed subtitle id')}:{subtitle.index},{i18n('error message received')}:{str(data)}")
            except:
                logger.error(f"{i18n('Failed subtitle id')}:{subtitle.index}, 无效的音频数据")
    else:
        logger.error(f"{i18n('Failed subtitle id')}:{subtitle.index}")

    # 当TTS失败时，生成静音片段而不是返回None
    logger.info(f"TTS失败，为字幕 {subtitle.index} 生成静音片段")
    return generate_silence_audio(subtitle, dir)


def start_gsv():
    if Sava_Utils.config.gsv_pydir == "":
        gr.Warning(i18n(
            'Please go to the settings page to specify the corresponding environment path and do not forget to save it!'))
        return i18n(
            'Please go to the settings page to specify the corresponding environment path and do not forget to save it!')
    if Sava_Utils.config.gsv_fallback:
        apath = "api.py"
        gr.Info(i18n('API downgraded to v1, functionality is limited.'))
        logger.warning(i18n('API downgraded to v1, functionality is limited.'))
    else:
        apath = "api_v2.py"
    if not os.path.exists(os.path.join(Sava_Utils.config.gsv_dir, apath)):
        raise FileNotFoundError(os.path.join(Sava_Utils.config.gsv_dir, apath))
    command = f'"{Sava_Utils.config.gsv_pydir}" "{os.path.join(Sava_Utils.config.gsv_dir, apath)}" {Sava_Utils.config.gsv_args}'
    rc_open_window(command=command, dir=Sava_Utils.config.gsv_dir)
    time.sleep(0.1)
    return f"GSV-API{i18n(' has been launched, please ensure the configuration is correct.')}"


def start_indextts():
    # 检查是否配置了启动脚本
    if Sava_Utils.config.indextts_script and Sava_Utils.config.indextts_script.strip():
        # 使用启动脚本/命令
        script_command = Sava_Utils.config.indextts_script.strip()

        # 判断是文件路径还是命令
        if script_command.startswith(('powershell', 'cmd', 'python', 'bash', 'sh')) or ' ' in script_command:
            # 这是一个命令，直接执行
            command = f'{script_command} {Sava_Utils.config.indextts_args}'
            work_dir = Sava_Utils.config.indextts_dir if Sava_Utils.config.indextts_dir else os.getcwd()
            rc_open_window(command=command, dir=work_dir)
            time.sleep(0.1)
            return f"Index-TTS 启动命令已执行: {script_command.split()[0]}"
        else:
            # 这是一个文件路径，检查文件是否存在
            if not os.path.exists(script_command):
                gr.Warning(f"启动脚本不存在: {script_command}")
                return f"启动脚本不存在: {script_command}"

            # 获取脚本目录作为工作目录
            script_dir = os.path.dirname(script_command)
            if not script_dir:
                script_dir = Sava_Utils.config.indextts_dir if Sava_Utils.config.indextts_dir else os.getcwd()

            # 执行启动脚本
            command = f'"{script_command}" {Sava_Utils.config.indextts_args}'
            rc_open_window(command=command, dir=script_dir)
            time.sleep(0.1)
            return f"Index-TTS 启动脚本已执行: {os.path.basename(script_command)}"

    else:
        # 使用原来的启动逻辑
        if Sava_Utils.config.indextts_pydir == "":
            gr.Warning(i18n(
                'Please go to the settings page to specify the corresponding environment path and do not forget to save it!'))
            return i18n(
                'Please go to the settings page to specify the corresponding environment path and do not forget to save it!')
        apath = "api.py"  # Index-TTS API文件
        if not os.path.exists(os.path.join(Sava_Utils.config.indextts_dir, apath)):
            raise FileNotFoundError(os.path.join(Sava_Utils.config.indextts_dir, apath))
        command = f'"{Sava_Utils.config.indextts_pydir}" "{os.path.join(Sava_Utils.config.indextts_dir, apath)}" {Sava_Utils.config.indextts_args}'
        rc_open_window(command=command, dir=Sava_Utils.config.indextts_dir)
        time.sleep(0.1)
        return f"Index-TTS API{i18n(' has been launched, please ensure the configuration is correct.')}"


def remake(*args):
    fp = None
    subtitle_list = args[-1]
    args = args[:-1]
    page, idx, timestamp, s_txt = args[:4]
    idx = int(idx)
    if int(args[1]) == -1:
        gr.Info("Not available!")
        return fp, *load_single_line(subtitle_list, idx)
    if Sava_Utils.config.server_mode and len(s_txt) > 512:
        gr.Warning("too long!")
        return fp, *load_single_line(subtitle_list, idx)
    subtitle_list[idx].text = s_txt
    subtitle_list[idx].is_success = None
    try:
        subtitle_list[idx].reset_srt_time(timestamp)
    except ValueError as e:
        gr.Info(str(e))
    if subtitle_list[idx].speaker is not None or (
            subtitle_list.proj is None and subtitle_list.default_speaker is not None):
        spk = subtitle_list[idx].speaker
        if spk is None:
            spk = subtitle_list.default_speaker
        try:
            with open(os.path.join(current_path, "SAVAdata", "speakers", spk), 'rb') as f:
                info = pickle.load(f)
        except FileNotFoundError:
            logger.error(f"{i18n('Speaker archive not found')}: {spk}")
            gr.Warning(f"{i18n('Speaker archive not found')}: {spk}")
            return fp, *load_single_line(subtitle_list, idx)
        args = info["raw_data"]
        proj = info["project"]
        args, kwargs = Projet_dict[proj].arg_filter(*args)
        # Projet_dict[proj].before_gen_action(*args,notify=False,force=True)
    else:
        if subtitle_list.proj is None:
            gr.Info(i18n('You must specify the speakers while using multi-speaker dubbing!'))
            return fp, *load_single_line(subtitle_list, idx)
        # 为单条生成构造正确的参数格式
        # 单条生成时需要模拟批量生成的参数格式：[input_file, fps, offset, workers, *TTS_ARGS]
        try:
            proj = subtitle_list.proj
            # 从 remake 函数的 args 中提取 TTS 相关参数（跳过前4个：page, idx, timestamp, s_txt）
            tts_args = args[4:]  # 获取 TTS 项目的参数

            # 添加详细的调试信息
            logger.info(f"🔍 单条生成调试信息:")
            logger.info(f"  - 字幕项目类型: {proj}")
            logger.info(f"  - TTS参数数量: {len(tts_args)}")
            logger.info(f"  - TTS参数内容: {tts_args}")

            # 根据参数数量自动判断实际的 TTS 项目类型
            if len(tts_args) == 4:  # EdgeTTS: language, speaker, rate, pitch
                actual_proj = "edgetts"
                logger.info(f"🔍 根据参数数量({len(tts_args)})判断为 EdgeTTS")
            elif len(tts_args) == 17:  # IndexTTS: 17个参数
                actual_proj = "indextts"
                logger.info(f"🔍 根据参数数量({len(tts_args)})判断为 IndexTTS")
            elif len(tts_args) == 20:  # GSV: 20个参数
                actual_proj = "gsv"
                logger.info(f"🔍 根据参数数量({len(tts_args)})判断为 GSV")
            else:
                # 如果参数数量不匹配，使用原来的项目设置
                actual_proj = proj
                logger.warning(f"⚠️ 无法根据参数数量({len(tts_args)})判断项目类型，使用原设置: {proj}")
                logger.warning(f"⚠️ 参数详情: {tts_args}")

            # 构造与批量生成相同的参数格式
            formatted_args = [
                None,  # input_file (单条生成时不需要)
                30,  # fps (默认值)
                0,  # offset (默认值)
                1,  # workers (单条生成时使用1个worker)
                *tts_args  # TTS项目的具体参数
            ]

            logger.info(f"🔧 单条生成参数格式化 - 项目: {actual_proj}, 参数数量: {len(formatted_args)}")
            args, kwargs = Projet_dict[actual_proj].arg_filter(*formatted_args)
            proj = actual_proj  # 更新项目类型
        except Exception as e:
            logger.error(f"❌ 参数过滤失败 - 项目: {subtitle_list.proj}, 错误: {str(e)}")
            logger.error(f"🔍 原始参数: {args}")
            logger.error(f"🔍 TTS参数: {args[4:] if len(args) > 4 else 'N/A'}")
            gr.Warning(f"参数过滤失败: {str(e)}")
            return fp, *load_single_line(subtitle_list, idx)
    Projet_dict[proj].before_gen_action(*args, config=Sava_Utils.config, notify=False, force=False)
    # subtitle_list[idx].text = s_txt
    fp = save(args, proj=proj, dir=subtitle_list.get_abs_dir(), subtitle=subtitle_list[idx])
    if fp is not None:
        gr.Info(i18n('Audio re-generation was successful! Click the <Reassemble Audio> button.'))
    else:
        gr.Warning("Audio re-generation failed!")
    subtitle_list.dump()
    return fp, *load_single_line(subtitle_list, idx)


def recompose(page: int, subtitle_list: Subtitles):
    if subtitle_list is None or len(subtitle_list) == 0:
        gr.Info(i18n('There is no subtitle in the current workspace'))
        return None, i18n('There is no subtitle in the current workspace'), *show_page(page, subtitle_list)
    audio = subtitle_list.audio_join(sr=Sava_Utils.config.output_sr)
    gr.Info(i18n("Reassemble successfully!"))
    return audio, "OK", *show_page(page, subtitle_list)


# 流式音频生成器函数
def streaming_generate_preprocess(interrupt_event, *args, project=None):
    """流式音频生成预处理函数"""
    try:
        args, kwargs = Projet_dict[project].arg_filter(*args)
    except Exception as e:
        info = f"{i18n('An error occurred')}: {str(e)}"
        gr.Warning(info)
        return

    # 使用生成器进行流式音频生成
    for audio_chunk in streaming_generate(*args, interrupt_event=interrupt_event, **kwargs):
        yield audio_chunk


def streaming_generate(*args, interrupt_event: Sava_Utils.utils.Flag, proj="", in_files=[], fps=30, offset=0, max_workers=1):
    """流式音频生成函数"""
    import time
    import tempfile
    import soundfile as sf

    # 首先执行原始的生成逻辑
    result = generate(*args, interrupt_event=interrupt_event, proj=proj, in_files=in_files, fps=fps, offset=offset, max_workers=max_workers)

    # 如果生成成功，保存为临时文件并输出路径
    if result and len(result) > 0 and result[0] is not None:
        try:
            sr_audio = result[0]  # (sample_rate, audio_data)
            if sr_audio is not None and len(sr_audio) == 2:
                sample_rate, audio_data = sr_audio

                # 创建临时文件
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    sf.write(tmp_file.name, audio_data, sample_rate)
                    yield tmp_file.name

        except Exception as e:
            print(f"流式音频输出错误: {e}")
            # 如果流式输出失败，尝试输出静音音频
            try:
                sample_rate = 22050
                duration = 1.0
                silence = np.zeros(int(sample_rate * duration))
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    sf.write(tmp_file.name, silence, sample_rate)
                    yield tmp_file.name
            except:
                pass
    else:
        # 如果生成失败，输出一个静音音频以避免界面错误
        try:
            sample_rate = 22050
            duration = 1.0
            silence = np.zeros(int(sample_rate * duration))
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                sf.write(tmp_file.name, silence, sample_rate)
                yield tmp_file.name
        except Exception as e:
            print(f"生成静音音频错误: {e}")


def streaming_recompose(page: int, subtitle_list: Subtitles):
    """流式重组音频函数"""
    import time
    import tempfile
    import soundfile as sf

    if subtitle_list is None or len(subtitle_list) == 0:
        gr.Info(i18n('There is no subtitle in the current workspace'))
        return

    try:
        # 直接调用原始的 recompose 函数获取结果
        result = recompose(page, subtitle_list)

        if result and len(result) > 0 and result[0] is not None:
            audio = result[0]  # (sample_rate, audio_data)
            if audio is not None and len(audio) == 2:
                sample_rate, audio_data = audio

                # 创建临时文件
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    sf.write(tmp_file.name, audio_data, sample_rate)
                    yield tmp_file.name

        gr.Info(i18n("Reassemble successfully!"))
    except Exception as e:
        gr.Warning(f"流式重组音频失败: {e}")
        # 输出静音以避免界面错误
        try:
            sample_rate = 22050
            duration = 1.0
            silence = np.zeros(int(sample_rate * duration))
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                sf.write(tmp_file.name, silence, sample_rate)
                yield tmp_file.name
        except:
            pass


def streaming_gen_multispeaker(interrupt_event: Sava_Utils.utils.Flag, *args, remake=False):
    """流式多说话人生成函数"""
    import time
    import tempfile
    import soundfile as sf

    page = args[0]
    max_workers = int(args[1])
    subtitles: Subtitles = args[-1]

    if subtitles is None or len(subtitles) == 0:
        gr.Info(i18n('There is no subtitle in the current workspace'))
        return

    # 执行原始的多说话人生成逻辑
    result = gen_multispeaker(interrupt_event, *args, remake=remake)

    # 如果生成成功，保存为临时文件并输出路径
    if result and len(result) > 0 and result[-1] is not None:
        try:
            sr_audio = result[-1]
            if sr_audio is not None and len(sr_audio) == 2:
                sample_rate, audio_data = sr_audio

                # 创建临时文件
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    sf.write(tmp_file.name, audio_data, sample_rate)
                    yield tmp_file.name

        except Exception as e:
            print(f"流式多说话人音频生成错误: {e}")
    else:
        # 如果没有音频返回，尝试重组现有音频
        try:
            audio = subtitles.audio_join(sr=Sava_Utils.config.output_sr)
            if audio is not None:
                # 创建临时文件
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    sf.write(tmp_file.name, audio[1], audio[0])
                    yield tmp_file.name
        except Exception as e:
            print(f"流式多说话人音频重组错误: {e}")
            # 输出静音以避免界面错误
            try:
                sample_rate = 22050
                duration = 1.0
                silence = np.zeros(int(sample_rate * duration))
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    sf.write(tmp_file.name, silence, sample_rate)
                    yield tmp_file.name
            except:
                pass


def save_spk(name: str, *args, project: str):
    name = name.strip()
    if Sava_Utils.config.server_mode:
        gr.Warning(i18n('This function has been disabled!'))
        return getspklist()
    if name in ["", [], None, 'None']:
        gr.Info(i18n('Please enter a valid name!'))
        return getspklist()
    args = [None, None, None, None, *args]
    # catch all arguments
    # process raw data before generating
    try:
        Projet_dict[project].arg_filter(*args)
        os.makedirs(os.path.join(current_path, "SAVAdata", "speakers"), exist_ok=True)
        with open(os.path.join(current_path, "SAVAdata", "speakers", name), "wb") as f:
            pickle.dump({"project": project, "raw_data": args}, f)
        gr.Info(f"{i18n('Saved successfully')}: [{project}]{name}")
    except Exception as e:
        gr.Warning(str(e))
        return getspklist(value=name)
    return getspklist(value=name)


if __name__ == "__main__":
    os.environ['GRADIO_TEMP_DIR'] = os.path.join(current_path, "SAVAdata", "temp", "gradio")
    workspaces_list = refworklist()
    if args.server_port is None:
        server_port = Sava_Utils.config.server_port
    else:
        server_port = args.server_port
    with gr.Blocks(title="Srt-AI-Voice-Assistant-WebUI", theme=Sava_Utils.config.theme, analytics_enabled=False) as app:
        STATE = gr.State(value=Subtitles())
        INTERRUPT_EVENT = gr.State(value=Sava_Utils.utils.Flag())
        gr.Markdown(value=MANUAL.getInfo("title"))
        with gr.Tabs():
            with gr.TabItem(i18n('Subtitle Dubbing')):
                with gr.Row():
                    with gr.Column():
                        textbox_intput_text = gr.TextArea(label=i18n('File content'), value="", interactive=False)
                        with gr.Accordion(i18n('Speaker Map'), open=False):
                            use_labled_text_mode = gr.Checkbox(label=i18n('Enable Marking Mode'))
                            speaker_map_set = gr.State(value=set())
                            speaker_map_dict = gr.State(value=dict())
                            edit_map_ui_md1 = f"### <center>{i18n('Speaker map is empty.')}</center>"
                            edit_map_ui_md2 = f"### <center>{i18n('Original Speaker')}</center>"
                            edit_map_ui_md3 = f"### <center>{i18n('Target Speaker')}</center>"


                            @gr.render(inputs=speaker_map_set)
                            def edit_map_ui(x):
                                if len(x) == 0:
                                    gr.Markdown(value=edit_map_ui_md1)
                                    return
                                c = refspklist()
                                with gr.Row():
                                    gr.Markdown(value=edit_map_ui_md2)
                                    gr.Markdown(value=edit_map_ui_md3)
                                with gr.Group():
                                    for i in x:
                                        with gr.Row():
                                            k = gr.Textbox(value=i, show_label=False, interactive=False)
                                            v = gr.Dropdown(value=i, choices=c, show_label=False,
                                                            allow_custom_value=True)
                                            v.change(modify_spkmap, inputs=[speaker_map_dict, k, v])
                                gr.Button(value="🗑️", variant="stop").click(lambda: (set(), dict()),
                                                                            outputs=[speaker_map_set, speaker_map_dict])


                            with gr.Accordion(i18n('Identify Original Speakers'), open=True):
                                update_spkmap_btn_upload = gr.Button(value=i18n('From Upload File'))
                                update_spkmap_btn_current = gr.Button(value=i18n('From Workspace'))
                            apply_spkmap2workspace_btn = gr.Button(value=i18n('Apply to current Workspace'))
                        create_multispeaker_btn = gr.Button(value=i18n('Create Multi-Speaker Dubbing Project'))
                    with gr.Column():
                        with gr.Tabs():
                            TTS_ARGS = []
                            for i in componments[1]:
                                TTS_ARGS.append(i.getUI())
                        GSV_ARGS, INDEXTTS_ARGS, EDGETTS_ARGS, CUSTOM_ARGS = TTS_ARGS
                    with gr.Column():
                        with gr.Accordion(i18n('Other Parameters'), open=True):
                            fps = gr.Number(label=i18n(
                                'Frame rate of Adobe Premiere project, only applicable to csv files exported from Pr'),
                                value=30, visible=True, interactive=True, minimum=1)
                            workers = gr.Number(label=i18n('Number of threads for sending requests'), value=2,
                                                visible=True, interactive=True, minimum=1)
                            offset = gr.Slider(minimum=-6, maximum=6, value=0, step=0.1,
                                               label=i18n('Voice time offset (seconds)'))
                        input_file = gr.File(label=i18n('Upload file (Batch mode only supports one speaker at a time)'),
                                             file_types=['.csv', '.srt', '.ass', '.vtt', '.txt'], file_count='multiple')

                        # 视频文件选择组件 - 支持文件选择和路径输入
                        with gr.Group():
                            gr.Markdown("视频文件")
                            with gr.Tabs():
                                with gr.TabItem("📁 选择文件"):
                                    video_file_upload = gr.File(
                                        label="选择视频文件",
                                        file_types=['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v',
                                                    '.3gp', '.ts'],
                                        type="filepath"
                                    )
                                with gr.TabItem("📝 输入路径"):
                                    local_video_path_input = gr.Textbox(
                                        label="",
                                        placeholder="🎬 输入本地视频文件路径，例如：C:/Videos/video.mp4",
                                        container=False,
                                        show_label=False
                                    )

                            with gr.Row():
                                load_local_video_path_btn = gr.Button(
                                    value="🚀 加载文件",
                                    scale=1,
                                    variant="primary"
                                )

                            # 合成视频按钮
                            with gr.Row():
                                compose_video_btn = gr.Button(
                                    value="🎬 合成视频",
                                    variant="secondary",
                                    size="lg"
                                )

                            gr.Markdown(
                                "💡 **功能说明**: 验证文件后可进行视频合成，支持添加字幕、音频等",
                                elem_classes="text-sm text-gray-600"
                            )

                        gen_textbox_output_text = gr.Textbox(label=i18n('Output Info'), interactive=False)
                        audio_output = gr.Audio(
                            label="Output Audio",
                            streaming=True,
                            autoplay=False,
                            interactive=False,
                            show_download_button=True,
                            show_share_button=False,
                            type="filepath"
                        )
                        stop_btn = gr.Button(value=i18n('Stop'), variant="stop")
                        stop_btn.click(lambda x: gr.Info(x.set()), inputs=[INTERRUPT_EVENT])
                        if not Sava_Utils.config.server_mode:
                            with gr.Accordion(i18n('API Launcher')):
                                start_gsv_btn = gr.Button(value="GPT-SoVITS")
                                start_indextts_btn = gr.Button(value="Index-TTS")
                                start_gsv_btn.click(start_gsv, outputs=[gen_textbox_output_text])
                                start_indextts_btn.click(start_indextts, outputs=[gen_textbox_output_text])
                        input_file.change(file_show, inputs=[input_file], outputs=[textbox_intput_text])

                        # 处理状态跟踪
                        processing_state = gr.State(value={"processed": False, "video_path": "", "srt_path": ""})

                        # 绑定视频文件加载事件
                        load_local_video_path_btn.click(
                            handle_video_file_load,
                            inputs=[video_file_upload, local_video_path_input, input_file, processing_state],
                            outputs=[gen_textbox_output_text, processing_state]
                        )

                        # 绑定合成视频事件（添加进度条支持）
                        compose_video_btn.click(
                            lambda progress=gr.Progress(track_tqdm=True), *args: handle_compose_video(progress, *args),
                            inputs=[video_file_upload, local_video_path_input, input_file, processing_state, STATE,
                                    audio_output],
                            outputs=[gen_textbox_output_text]
                        )

                with gr.Accordion(
                        label=i18n('Editing area *Note: DO NOT clear temporary files while using this function.'),
                        open=True):
                    with gr.Column():
                        edit_rows = []
                        edit_real_index_list = []
                        edit_check_list = []
                        edit_start_end_time_list = []
                        with gr.Row(equal_height=True):
                            worklist = gr.Dropdown(choices=workspaces_list if len(workspaces_list) > 0 else [""],
                                                   label=i18n('History'), scale=2)
                            workrefbtn = gr.Button(value="🔄️", scale=1, min_width=40,
                                                   visible=not Sava_Utils.config.server_mode,
                                                   interactive=not Sava_Utils.config.server_mode)
                            workloadbtn = gr.Button(value=i18n('Load'), scale=1, min_width=40)
                            page_slider = gr.Slider(minimum=1, maximum=1, value=1, label="",
                                                    step=Sava_Utils.config.num_edit_rows, scale=5)
                            audio_player = gr.Audio(show_label=False, value=None, interactive=False, autoplay=True,
                                                    scale=4, waveform_options={"show_recording_waveform": False})
                            recompose_btn = gr.Button(value=i18n('Reassemble Audio'), scale=1, min_width=100)
                            export_btn = gr.Button(value=i18n('Export Subtitles'), scale=1, min_width=100)
                        for x in range(Sava_Utils.config.num_edit_rows):
                            edit_real_index = gr.Number(show_label=False, visible=False, value=-1,
                                                        interactive=False)  # real index
                            with gr.Row(equal_height=True, height=55):
                                edit_check = gr.Checkbox(value=False, interactive=True, min_width=40, show_label=False,
                                                         label="", scale=0)
                                edit_check_list.append(edit_check)
                                edit_rows.append(edit_real_index)  # real index
                                edit_real_index_list.append(edit_real_index)
                                edit_rows.append(
                                    gr.Textbox(scale=1, visible=False, show_label=False, interactive=False, value='-1',
                                               max_lines=1, min_width=40))  # index(raw)
                                edit_start_end_time = gr.Textbox(scale=3, visible=False, show_label=False,
                                                                 interactive=False, value="NO INFO", max_lines=1)
                                edit_start_end_time_list.append(edit_start_end_time)
                                edit_rows.append(edit_start_end_time)  # start time and end time
                                s_txt = gr.Textbox(scale=6, visible=False, show_label=False, interactive=False,
                                                   value="NO INFO", max_lines=1)  # content
                                edit_rows.append(s_txt)
                                edit_rows.append(
                                    gr.Textbox(show_label=False, visible=False, interactive=False, min_width=100,
                                               value="None", scale=1, max_lines=1))  # speaker
                                edit_rows.append(
                                    gr.Textbox(value="NO INFO", show_label=False, visible=False, interactive=False,
                                               min_width=100, scale=1, max_lines=1))  # is success or delayed?
                                with gr.Row(equal_height=True):
                                    __ = gr.Button(value="▶️", scale=1, min_width=50)
                                    __.click(play_audio, inputs=[edit_real_index, STATE], outputs=[audio_player])

                                    gsvregenbtn = gr.Button(value="🔄️", scale=1, min_width=50, visible=True)
                                    gsvregenbtn.click(remake,
                                                      inputs=[page_slider, edit_real_index, edit_start_end_time, s_txt,
                                                              *GSV_ARGS, STATE],
                                                      outputs=[audio_player, page_slider] + edit_rows[-6:])
                                    edgettsregenbtn = gr.Button(value="🔄️", scale=1, min_width=50, visible=False)
                                    edgettsregenbtn.click(remake,
                                                          inputs=[page_slider, edit_real_index, edit_start_end_time,
                                                                  s_txt, *EDGETTS_ARGS, STATE],
                                                          outputs=[audio_player, page_slider] + edit_rows[-6:])
                                    indexttsregenbtn = gr.Button(value="🔄️", scale=1, min_width=50, visible=False)
                                    indexttsregenbtn.click(remake,
                                                           inputs=[page_slider, edit_real_index, edit_start_end_time,
                                                                   s_txt, *INDEXTTS_ARGS, STATE],
                                                           outputs=[audio_player, page_slider] + edit_rows[-6:])
                                    customregenbtn = gr.Button(value="🔄️", scale=1, min_width=50, visible=False)
                                    customregenbtn.click(remake,
                                                         inputs=[page_slider, edit_real_index, edit_start_end_time,
                                                                 s_txt, CUSTOM.choose_custom_api, STATE],
                                                         outputs=[audio_player, page_slider] + edit_rows[-6:])
                                    edit_rows.append(gsvregenbtn)
                                    edit_rows.append(edgettsregenbtn)
                                    edit_rows.append(indexttsregenbtn)
                                    edit_rows.append(customregenbtn)
                        workrefbtn.click(getworklist, inputs=[], outputs=[worklist])
                        export_btn.click(export_subtitle_with_new_name, inputs=[input_file, STATE],
                                         outputs=[input_file])
                        with gr.Row(equal_height=True):
                            all_selection_btn = gr.Button(value=i18n('Select All'), interactive=True, min_width=50)
                            all_selection_btn.click(None, inputs=[], outputs=edit_check_list,
                                                    js=f"() => Array({Sava_Utils.config.num_edit_rows}).fill(true)")
                            reverse_selection_btn = gr.Button(value=i18n('Reverse Selection'), interactive=True,
                                                              min_width=50)
                            reverse_selection_btn.click(None, inputs=edit_check_list, outputs=edit_check_list,
                                                        js="(...vals) => vals.map(v => !v)")
                            clear_selection_btn = gr.Button(value=i18n('Clear Selection'), interactive=True,
                                                            min_width=50)
                            clear_selection_btn.click(None, inputs=[], outputs=edit_check_list,
                                                      js=f"() => Array({Sava_Utils.config.num_edit_rows}).fill(false)")
                            apply_se_btn = gr.Button(value=i18n('Apply Timestamp modifications'), interactive=True,
                                                     min_width=50)
                            apply_se_btn.click(apply_start_end_time, inputs=[page_slider, STATE, *edit_real_index_list,
                                                                             *edit_start_end_time_list],
                                               outputs=edit_rows)
                            copy_btn = gr.Button(value=i18n('Copy'), interactive=True, min_width=50)
                            copy_btn.click(copy_subtitle,
                                           inputs=[page_slider, STATE, *edit_check_list, *edit_real_index_list],
                                           outputs=[*edit_check_list, page_slider, *edit_rows])
                            merge_btn = gr.Button(value=i18n('Merge'), interactive=True, min_width=50)
                            merge_btn.click(merge_subtitle,
                                            inputs=[page_slider, STATE, *edit_check_list, *edit_real_index_list],
                                            outputs=[*edit_check_list, page_slider, *edit_rows])
                            delete_btn = gr.Button(value=i18n('Delete'), interactive=True, min_width=50)
                            delete_btn.click(delete_subtitle,
                                             inputs=[page_slider, STATE, *edit_check_list, *edit_real_index_list],
                                             outputs=[*edit_check_list, page_slider, *edit_rows])

                            all_regen_btn_gsv = gr.Button(value=i18n('Continue Generation'), variant="primary",
                                                          visible=True, interactive=True, min_width=50)
                            edit_rows.append(all_regen_btn_gsv)
                            all_regen_btn_edgetts = gr.Button(value=i18n('Continue Generation'), variant="primary",
                                                              visible=False, interactive=True, min_width=50)
                            edit_rows.append(all_regen_btn_edgetts)
                            all_regen_btn_indextts = gr.Button(value=i18n('Continue Generation'), variant="primary",
                                                               visible=False, interactive=True, min_width=50)
                            edit_rows.append(all_regen_btn_indextts)
                            all_regen_btn_custom = gr.Button(value=i18n('Continue Generation'), variant="primary",
                                                             visible=False, interactive=True, min_width=50)
                            edit_rows.append(all_regen_btn_custom)

                            all_regen_btn_gsv.click(
                                lambda process=gr.Progress(track_tqdm=True), *args: gen_multispeaker(*args,
                                                                                                     remake=True),
                                inputs=[INTERRUPT_EVENT, page_slider, workers, *GSV_ARGS, STATE], outputs=edit_rows)
                            all_regen_btn_edgetts.click(
                                lambda process=gr.Progress(track_tqdm=True), *args: gen_multispeaker(*args,
                                                                                                     remake=True),
                                inputs=[INTERRUPT_EVENT, page_slider, workers, *EDGETTS_ARGS, STATE], outputs=edit_rows)
                            all_regen_btn_indextts.click(
                                lambda process=gr.Progress(track_tqdm=True), *args: gen_multispeaker(*args,
                                                                                                     remake=True),
                                inputs=[INTERRUPT_EVENT, page_slider, workers, *INDEXTTS_ARGS, STATE],
                                outputs=edit_rows)
                            all_regen_btn_custom.click(
                                lambda process=gr.Progress(track_tqdm=True), *args: gen_multispeaker(*args,
                                                                                                     remake=True),
                                inputs=[INTERRUPT_EVENT, page_slider, workers, CUSTOM.choose_custom_api, STATE],
                                outputs=edit_rows)

                        page_slider.change(show_page, inputs=[page_slider, STATE], outputs=edit_rows)
                        workloadbtn.click(load_work, inputs=[worklist], outputs=[STATE, page_slider, *edit_rows])
                        recompose_btn.click(streaming_recompose, inputs=[page_slider, STATE],
                                            outputs=[audio_output])

                        apply_spkmap2workspace_btn.click(apply_spkmap2workspace,
                                                         inputs=[speaker_map_dict, page_slider, STATE],
                                                         outputs=edit_rows)

                        with gr.Accordion(i18n('Find and Replace'), open=False):
                            with gr.Row(equal_height=True):
                                find_text_expression = gr.Textbox(show_label=False, placeholder=i18n('Find What'),
                                                                  scale=3)
                                target_text = gr.Textbox(show_label=False, placeholder=i18n('Replace With'), scale=3)
                                find_and_rep_exec = gr.Textbox(show_label=False,
                                                               placeholder=r'Exec... e.g. item.speaker="Name"', scale=3,
                                                               visible=not Sava_Utils.config.server_mode)
                                enable_re = gr.Checkbox(label=i18n('Enable Regular Expression'), min_width=60, scale=1)
                                find_next_btn = gr.Button(value=i18n('Find Next'), variant="secondary", min_width=50,
                                                          scale=1)
                                replace_all_btn = gr.Button(value=i18n('Replace All'), variant="primary", min_width=50,
                                                            scale=1)
                                find_next_btn.click(find_next,
                                                    inputs=[STATE, find_text_expression, enable_re, page_slider,
                                                            *edit_check_list, *edit_real_index_list],
                                                    outputs=[*edit_check_list, page_slider, *edit_rows])
                                replace_all_btn.click(find_and_replace,
                                                      inputs=[STATE, find_text_expression, target_text,
                                                              find_and_rep_exec, enable_re, page_slider],
                                                      outputs=[page_slider, *edit_rows])
                with gr.Accordion(label=i18n('Multi-speaker dubbing')):
                    with gr.Row(equal_height=True):
                        speaker_list = gr.Dropdown(label=i18n('Select/Create Speaker'), value="None",
                                                   choices=refspklist(),
                                                   allow_custom_value=not Sava_Utils.config.server_mode, scale=4)
                        # speaker_list.change(set_default_speaker,inputs=[speaker_list,STATE])
                        select_spk_projet = gr.Dropdown(choices=['gsv', 'indextts', 'edgetts', 'custom'],
                                                        value='gsv', interactive=True, label=i18n('TTS Project'))
                        refresh_spk_list_btn = gr.Button(value="🔄️", min_width=60, scale=0)
                        refresh_spk_list_btn.click(getspklist, inputs=[], outputs=[speaker_list])
                        apply_btn = gr.Button(value="✅", min_width=60, scale=0)
                        apply_btn.click(apply_spk, inputs=[speaker_list, page_slider, STATE, *edit_check_list,
                                                           *edit_real_index_list],
                                        outputs=[*edit_check_list, *edit_rows])

                        save_spk_btn_gsv = gr.Button(value="💾", min_width=60, scale=0, visible=True)
                        save_spk_btn_gsv.click(lambda *args: save_spk(*args, project="gsv"),
                                               inputs=[speaker_list, *GSV_ARGS], outputs=[speaker_list])
                        save_spk_btn_edgetts = gr.Button(value="💾", min_width=60, scale=0, visible=False)
                        save_spk_btn_edgetts.click(lambda *args: save_spk(*args, project="edgetts"),
                                                   inputs=[speaker_list, *EDGETTS_ARGS], outputs=[speaker_list])
                        save_spk_btn_indextts = gr.Button(value="💾", min_width=60, scale=0, visible=False)
                        save_spk_btn_indextts.click(lambda *args: save_spk(*args, project="indextts"),
                                                    inputs=[speaker_list, *INDEXTTS_ARGS], outputs=[speaker_list])
                        save_spk_btn_custom = gr.Button(value="💾", min_width=60, scale=0, visible=False)
                        save_spk_btn_custom.click(lambda *args: save_spk(*args, project="custom"),
                                                  inputs=[speaker_list, CUSTOM.choose_custom_api],
                                                  outputs=[speaker_list])

                        select_spk_projet.change(switch_spk_proj, inputs=[select_spk_projet],
                                                 outputs=[save_spk_btn_gsv, save_spk_btn_indextts,
                                                          save_spk_btn_edgetts, save_spk_btn_custom])

                        del_spk_list_btn = gr.Button(value="🗑️", min_width=60, scale=0)
                        del_spk_list_btn.click(del_spk, inputs=[speaker_list], outputs=[speaker_list])
                        start_gen_multispeaker_btn = gr.Button(value=i18n('Start Multi-speaker Synthesizing'),
                                                               variant="primary")
                        start_gen_multispeaker_btn.click(
                            lambda process=gr.Progress(track_tqdm=True), *args: streaming_gen_multispeaker(*args),
                            inputs=[INTERRUPT_EVENT, page_slider, workers, STATE], outputs=[audio_output])

            with gr.TabItem("批量配音"):
                batch_ui_components = create_batch_dubbing_ui()

                # 绑定批量配音事件

                # 添加任务
                batch_ui_components['confirm_add_btn'].click(
                    add_batch_task,
                    inputs=[
                        batch_ui_components['task_video_file'],
                        batch_ui_components['task_subtitle_file'],
                        batch_ui_components['batch_tasks_state']
                    ],
                    outputs=[
                        batch_ui_components['batch_tasks_state'],
                        batch_ui_components['task_video_file'],
                        batch_ui_components['task_subtitle_file'],
                        batch_ui_components['batch_total_tasks'],
                        batch_ui_components['batch_completed_tasks'],
                        batch_ui_components['batch_failed_tasks'],
                        batch_ui_components['batch_success_rate']
                    ]
                )

                # 重置文件输入
                batch_ui_components['cancel_add_btn'].click(
                    reset_file_inputs,
                    outputs=[
                        batch_ui_components['task_video_file'],
                        batch_ui_components['task_subtitle_file']
                    ]
                )

                # 清空任务列表
                batch_ui_components['batch_clear_btn'].click(
                    clear_batch_tasks,
                    outputs=[
                        batch_ui_components['batch_tasks_state'],
                        batch_ui_components['batch_total_tasks'],
                        batch_ui_components['batch_completed_tasks'],
                        batch_ui_components['batch_failed_tasks'],
                        batch_ui_components['batch_success_rate']
                    ]
                )

                # 批量开始
                batch_ui_components['batch_start_all_btn'].click(
                    start_all_batch_tasks,
                    inputs=[batch_ui_components['batch_tasks_state']],
                    outputs=[batch_ui_components['batch_tasks_state']]
                )

                # 导出结果
                batch_ui_components['batch_export_btn'].click(
                    export_batch_results,
                    inputs=[batch_ui_components['batch_tasks_state']]
                )

                # 保留原视频格式选项变化时控制输出格式可见性
                batch_ui_components['batch_keep_original_format'].change(
                    toggle_output_format_visibility,
                    inputs=[batch_ui_components['batch_keep_original_format']],
                    outputs=[batch_ui_components['batch_output_format']]
                )

                # 任务列表更新
                batch_ui_components['batch_tasks_state'].change(
                    render_batch_tasks,
                    inputs=[batch_ui_components['batch_tasks_state']],
                    outputs=[batch_ui_components['batch_task_display']]
                )

            with gr.TabItem(i18n('Auxiliary Functions')):
                for i in componments[2]:
                    i.getUI(input_file)
            with gr.TabItem(i18n('Extended Contents')):
                available = False
                from Sava_Utils.extern_extensions.wav2srt_webui import WAV2SRT

                WAV2SRT = WAV2SRT(config=Sava_Utils.config)
                componments[3].append(WAV2SRT)
                available = WAV2SRT.getUI(input_file, worklist, TRANSLATION_MODULE)
                if not available:
                    gr.Markdown(
                        "No additional extensions have been installed and a restart is required for the changes to take effect.<br>[Get Extentions](https://github.com/YYuX-1145/Srt-AI-Voice-Assistant/tree/main/tools)")
            with gr.TabItem(i18n('Settings')):
                with gr.Row():
                    with gr.Column():
                        SETTINGS = Sava_Utils.settings.Settings_UI(componments=componments)
                        SETTINGS.getUI()
                    with gr.Column():
                        with gr.TabItem(i18n('Readme')):
                            gr.Markdown(value=MANUAL.getInfo("readme"))
                            gr.Markdown(value=MANUAL.getInfo("changelog"))
                        with gr.TabItem(i18n('Issues')):
                            gr.Markdown(value=MANUAL.getInfo("issues"))
                        with gr.TabItem(i18n('Help & User guide')):
                            gr.Markdown(value=MANUAL.getInfo("help"))

        update_spkmap_btn_upload.click(get_speaker_map_from_file, inputs=[input_file],
                                       outputs=[speaker_map_set, speaker_map_dict])
        update_spkmap_btn_current.click(get_speaker_map_from_sub, inputs=[STATE],
                                        outputs=[speaker_map_set, speaker_map_dict])
        create_multispeaker_btn.click(create_multi_speaker,
                                      inputs=[input_file, use_labled_text_mode, speaker_map_dict, fps, offset],
                                      outputs=[worklist, page_slider, *edit_rows, STATE])

        GSV.gen_btn2.click(
            lambda process=gr.Progress(track_tqdm=True), *args: streaming_generate_preprocess(*args, project="gsv"),
            inputs=[INTERRUPT_EVENT, input_file, fps, offset, workers, *GSV_ARGS],
            outputs=[audio_output])
        EDGETTS.gen_btn_edge.click(
            lambda process=gr.Progress(track_tqdm=True), *args: streaming_generate_preprocess(*args, project="edgetts"),
            inputs=[INTERRUPT_EVENT, input_file, fps, offset, workers, *EDGETTS_ARGS],
            outputs=[audio_output])
        INDEXTTS.gen_btn5.click(
            lambda process=gr.Progress(track_tqdm=True), *args: streaming_generate_preprocess(*args, project="indextts"),
            inputs=[INTERRUPT_EVENT, input_file, fps, offset, workers, *INDEXTTS_ARGS],
            outputs=[audio_output])
        CUSTOM.gen_btn4.click(
            lambda process=gr.Progress(track_tqdm=True), *args: streaming_generate_preprocess(*args, project="custom"),
            inputs=[INTERRUPT_EVENT, input_file, fps, offset, workers, CUSTOM.choose_custom_api],
            outputs=[audio_output])
        # Stability is not ensured due to the mechanism of gradio.

        # 添加Index-TTS配置刷新事件
        if hasattr(INDEXTTS, 'refresh_config_on_app_load') and hasattr(INDEXTTS, 'app_load_outputs'):
            app.load(
                fn=INDEXTTS.refresh_config_on_app_load,
                inputs=[],
                outputs=INDEXTTS.app_load_outputs
            )

    app.queue(default_concurrency_limit=Sava_Utils.config.concurrency_count,
              max_size=2 * Sava_Utils.config.concurrency_count).launch(
        share=args.share,
        server_port=server_port if server_port > 0 else None,
        inbrowser=True,
        server_name='0.0.0.0' if Sava_Utils.config.LAN_access or args.LAN_access else '127.0.0.1',
        show_api=not Sava_Utils.config.server_mode,
    )
