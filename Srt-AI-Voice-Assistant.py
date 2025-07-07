import io
import os
import sys
# import inspect
import warnings

if getattr(sys, "frozen", False):
    current_path = os.path.dirname(sys.executable)
    os.environ["exe"] = 'True'
elif __file__:
    current_path = os.path.dirname(__file__)
    os.environ["exe"] = 'False'
os.environ["current_path"] = current_path

warnings.filterwarnings("ignore", category=UserWarning)

import json

import datetime
import soundfile as sf
import concurrent.futures
from tqdm import tqdm
from collections import defaultdict

from Sava_Utils import args, MANUAL
from Sava_Utils.utils import *
from Sava_Utils.edit_panel import *
from Sava_Utils.subtitle import Subtitle, Subtitles
from Sava_Utils.video_speed_adjuster import adjust_video_speed_by_subtitles

import Sava_Utils.tts_projects
import Sava_Utils.tts_projects.bv2
import Sava_Utils.tts_projects.gsv
import Sava_Utils.tts_projects.edgetts
import Sava_Utils.tts_projects.custom
import Sava_Utils.tts_projects.indextts
from Sava_Utils.subtitle_translation import Translation_module
from Sava_Utils.polyphone import Polyphone

BV2 = Sava_Utils.tts_projects.bv2.BV2(Sava_Utils.config)
GSV = Sava_Utils.tts_projects.gsv.GSV(Sava_Utils.config)
EDGETTS = Sava_Utils.tts_projects.edgetts.EdgeTTS(Sava_Utils.config)
CUSTOM = Sava_Utils.tts_projects.custom.Custom(Sava_Utils.config)
INDEXTTS = Sava_Utils.tts_projects.indextts.IndexTTS(Sava_Utils.config)
TRANSLATION_MODULE = Translation_module(Sava_Utils.config)
POLYPHONE = Polyphone(Sava_Utils.config)
Projet_dict = {"bv2": BV2, "gsv": GSV, "edgetts": EDGETTS, "indextts": INDEXTTS, "custom": CUSTOM}
componments = {
    1: [GSV, BV2, INDEXTTS, EDGETTS, CUSTOM],
    2: [TRANSLATION_MODULE, POLYPHONE],
    3: [],
}


def custom_api(text):
    raise i18n('You need to load custom API functions!')


def export_subtitle_with_new_name(file_list, subtitle_state):
    """
    导出字幕文件，基于原字幕文件名生成新的文件名

    Args:
        file_list: 原始文件列表
        subtitle_state: 字幕状态对象

    Returns:
        更新后的文件列表
    """
    try:
        # 获取原始文件列表
        original_files = [i.name for i in file_list] if file_list else []

        # 导出字幕文件
        exported_file = subtitle_state.export()

        if exported_file:
            # 如果有原始字幕文件，基于第一个文件名生成新名称
            if original_files:
                original_file = original_files[0]
                # 获取原文件的目录和基础名称
                original_dir = os.path.dirname(original_file)
                original_basename = Sava_Utils.utils.basename_no_ext(original_file)

                new_filename = f"{original_basename}.srt"

                # 如果原文件在output目录外，则放到output目录
                if "SAVAdata" not in original_dir or "output" not in original_dir:
                    new_filepath = os.path.join(current_path, "SAVAdata", "output", new_filename)
                else:
                    new_filepath = os.path.join(original_dir, new_filename)

                # 如果新文件名与导出文件名不同，则重命名
                if exported_file != new_filepath:
                    try:
                        # 确保目标目录存在
                        os.makedirs(os.path.dirname(new_filepath), exist_ok=True)

                        # 如果目标文件已存在，添加时间戳
                        if os.path.exists(new_filepath):
                            timestamp = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
                            name_part = Sava_Utils.utils.basename_no_ext(new_filepath)
                            new_filepath = os.path.join(os.path.dirname(new_filepath), f"{name_part}{timestamp}.srt")

                        # 重命名文件
                        shutil.move(exported_file, new_filepath)
                        exported_file = new_filepath

                        print(f"✅ 字幕文件已导出: {new_filepath}")
                        gr.Info(f"字幕文件已导出: {os.path.basename(new_filepath)}")

                    except Exception as e:
                        print(f"⚠️ 重命名文件失败: {e}")
                        gr.Warning(f"重命名文件失败，使用默认名称: {os.path.basename(exported_file)}")

            # 返回原始文件列表 + 新导出的文件
            return original_files + [exported_file]
        else:
            # 如果导出失败，返回原始文件列表
            return original_files

    except Exception as e:
        print(f"❌ 导出字幕文件失败: {e}")
        gr.Error(f"导出字幕文件失败: {str(e)}")
        return [i.name for i in file_list] if file_list else []


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
            data = json.loads(audio)
            logger.error(f"{i18n('Failed subtitle id')}:{subtitle.index},{i18n('error message received')}:{str(data)}")
            subtitle.is_success = False
            return None
    else:
        logger.error(f"{i18n('Failed subtitle id')}:{subtitle.index}")
        subtitle.is_success = False
        return None


def start_hiyoriui():
    if Sava_Utils.config.bv2_pydir == "":
        gr.Warning(i18n(
            'Please go to the settings page to specify the corresponding environment path and do not forget to save it!'))
        return i18n(
            'Please go to the settings page to specify the corresponding environment path and do not forget to save it!')
    command = f'"{Sava_Utils.config.bv2_pydir}" "{os.path.join(Sava_Utils.config.bv2_dir, "hiyoriUI.py")}" {Sava_Utils.config.bv2_args}'
    rc_open_window(command=command, dir=Sava_Utils.config.bv2_dir)
    time.sleep(0.1)
    return f"HiyoriUI{i18n(' has been launched, please ensure the configuration is correct.')}"


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
        # args = [None, *args]  # ~~fill data~~
        try:
            proj = subtitle_list.proj
            args, kwargs = Projet_dict[proj].arg_filter(*args)
        except Exception as e:
            # print(e)
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
                        TTS_ARGS = []
                        for i in componments[1]:
                            TTS_ARGS.append(i.getUI())
                    GSV_ARGS, BV2_ARGS, INDEXTTS_ARGS, EDGETTS_ARGS, CUSTOM_ARGS = TTS_ARGS
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
                                             file_types=['.csv', '.srt', '.txt'], file_count='multiple')

                        # 本地视频地址输入组件 - 优化样式
                        with gr.Group():
                            gr.Markdown("视频文件路径")
                            with gr.Row():
                                local_video_path_input = gr.Textbox(
                                    label="",
                                    placeholder="🎬 输入本地视频文件路径，例如：C:/Videos/video.mp4",
                                    scale=4,
                                    container=False,
                                    show_label=False
                                )
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
                        audio_output = gr.Audio(label="Output Audio")
                        stop_btn = gr.Button(value=i18n('Stop'), variant="stop")
                        stop_btn.click(lambda x: gr.Info(x.set()), inputs=[INTERRUPT_EVENT])
                        if not Sava_Utils.config.server_mode:
                            with gr.Accordion(i18n('API Launcher')):
                                start_hiyoriui_btn = gr.Button(value="HiyoriUI")
                                start_gsv_btn = gr.Button(value="GPT-SoVITS")
                                start_hiyoriui_btn.click(start_hiyoriui, outputs=[gen_textbox_output_text])
                                start_gsv_btn.click(start_gsv, outputs=[gen_textbox_output_text])
                        input_file.change(file_show, inputs=[input_file], outputs=[textbox_intput_text])

                        # 处理状态跟踪
                        processing_state = gr.State(value={"processed": False, "video_path": "", "srt_path": ""})


                        # 本地视频路径处理函数
                        def handle_local_video_path_load(video_path, uploaded_files, current_state):
                            """处理本地视频路径加载和音频分离"""
                            if not video_path or video_path.strip() == "":
                                return gr.update(
                                    value="⚠️ **请输入视频文件路径**\n\n💡 示例路径：`C:/Videos/movie.mp4`"), current_state

                            # 清理路径（移除引号和多余空格）
                            video_path = video_path.strip().strip('"').strip("'")

                            # 检查文件是否存在
                            if not os.path.exists(video_path):
                                return gr.update(
                                    value=f"❌ **文件不存在**\n\n📂 检查路径：`{video_path}`\n\n💡 请确认文件路径是否正确"), current_state

                            # 检查是否是文件（不是目录）
                            if not os.path.isfile(video_path):
                                return gr.update(
                                    value=f"❌ **这是一个目录，不是文件**\n\n📂 路径：`{video_path}`\n\n💡 请选择具体的视频文件"), current_state

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

                            # 获取字幕文件路径
                            srt_file = None
                            for file in uploaded_files:
                                if file.name.lower().endswith('.srt'):
                                    srt_file = file.name
                                    break

                            if not srt_file:
                                return gr.update(
                                    value="⚠️ **未找到字幕文件**\n\n📝 上传的文件中没有 .srt 格式的字幕文件\n\n💡 请上传正确的字幕文件"), current_state

                            # 检查是否已经处理过相同的文件
                            if (current_state["processed"] and
                                    current_state["video_path"] == video_path and
                                    current_state["srt_path"] == srt_file):
                                return gr.update(
                                    value="ℹ️ **文件已处理过**\n\n✅ 相同的视频和字幕文件已经处理过了\n\n💡 如需重新处理，请更换文件或重启程序"), current_state

                            try:
                                # 获取文件信息
                                file_size = os.path.getsize(video_path)
                                file_size_mb = file_size / (1024 * 1024)
                                file_name = os.path.basename(video_path)

                                # 检查文件权限
                                if not os.access(video_path, os.R_OK):
                                    return gr.update(
                                        value=f"❌ **文件权限不足**\n\n🔒 无法读取文件：`{video_path}`\n\n💡 请检查文件权限或以管理员身份运行"), current_state

                                # 开始处理
                                feedback_message = f"""
🚀 **开始处理视频文件**

📋 **文件信息**
• 📁 视频文件: `{file_name}`
• 📏 文件大小: **{file_size_mb:.1f} MB**
• 🎞️ 视频格式: **{file_extension.upper()}**
• 📝 字幕文件: `{os.path.basename(srt_file)}`

⏳ **正在执行以下步骤:**
1. 🎵 从视频中分离音频 (separate_video_audio)
2. ✂️ 根据字幕分割音频 (split_audio_by_subtitles)

请稍候...
                                """.strip()

                                # 导入音频分离模块
                                import sys
                                sys.path.insert(0, 'Sava_Utils')
                                import audio_separator

                                # 步骤1: 分离视频音频
                                # 生成唯一的哈希目录名（基于视频路径和字幕路径）
                                import hashlib
                                hash_input = f"{video_path}_{srt_file}_{time.time()}"
                                session_hash = hashlib.sha256(hash_input.encode()).hexdigest()

                                # 使用项目标准的存储路径，包含哈希子目录
                                base_temp_dir = os.path.join(current_path, "SAVAdata", "temp")
                                output_dir = os.path.join(base_temp_dir, "audio_processing", session_hash)
                                os.makedirs(output_dir, exist_ok=True)

                                result = audio_separator.separate_video_audio(video_path, output_dir)

                                # 使用人声音频进行分割
                                vocal_audio_path = result.get('vocal_audio')
                                if not vocal_audio_path or not os.path.exists(vocal_audio_path):
                                    return gr.update(
                                        value="❌ **音频分离失败**\n\n🔧 无法生成人声音频文件\n\n💡 请检查视频文件是否包含音频轨道"), current_state

                                # 步骤2: 根据字幕分割音频
                                segments_dir = os.path.join(output_dir, "segments")
                                os.makedirs(segments_dir, exist_ok=True)
                                segments = audio_separator.split_audio_by_subtitles(vocal_audio_path, srt_file,
                                                                                    segments_dir)

                                # 更新处理状态，保存所有处理结果
                                new_state = {
                                    "processed": True,
                                    "video_path": video_path,
                                    "srt_path": srt_file,
                                    "processing_result": result,  # 保存完整的处理结果
                                    "session_hash": session_hash,
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
• 🎬 视频文件: `SAVAdata/temp/audio_processing/{session_hash[:8]}...`
• ✂️ 音频片段: `SAVAdata/temp/audio_processing/{session_hash[:8]}.../segments/`

🔑 **会话ID**: `{session_hash[:16]}...`

🎯 文件已隔离保存，不同操作结果互不干扰！
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


                        # 绑定本地视频路径加载事件
                        load_local_video_path_btn.click(
                            handle_local_video_path_load,
                            inputs=[local_video_path_input, input_file, processing_state],
                            outputs=[gen_textbox_output_text, processing_state]
                        )


                        # 合成视频处理函数
                        def handle_compose_video(video_path, subtitle_files, current_state, subtitles_state, audio_data):
                            """处理视频合成 - 完整检查版本"""

                            # 1. 检查字幕是否上传
                            if not subtitle_files or len(subtitle_files) == 0:
                                return gr.update(
                                    value="❌ **字幕文件检查失败**\n\n📝 **错误**: 未上传字幕文件\n\n💡 **解决方案**: 请在左侧上传 .srt 格式的字幕文件")

                            # 检查字幕文件格式
                            srt_files = [f for f in subtitle_files if f.name.lower().endswith('.srt')]
                            if len(srt_files) == 0:
                                return gr.update(
                                    value="❌ **字幕文件格式错误**\n\n📝 **错误**: 上传的文件中没有 .srt 格式的字幕文件\n\n💡 **解决方案**: 请上传正确的 .srt 字幕文件")

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
                                file_name = os.path.basename(video_path)
                                file_size = os.path.getsize(video_path)
                                file_size_mb = file_size / (1024 * 1024)
                                file_extension = os.path.splitext(video_path)[1].lower()

                                # 步骤1: 导出字幕文件
                                # 创建临时目录用于视频处理
                                temp_dir = os.path.join(current_path, "SAVAdata", "temp", "video_compose")
                                os.makedirs(temp_dir, exist_ok=True)

                                # 创建输出目录用于最终文件
                                output_dir = os.path.join(current_path, "SAVAdata", "output")
                                os.makedirs(output_dir, exist_ok=True)

                                # 生成基于项目名称的文件名
                                project_name = subtitles_state.dir if subtitles_state.dir else "video_compose"

                                # 导出原始字幕到临时目录（用于视频处理）
                                original_srt_path = os.path.join(temp_dir, "original.srt")
                                shutil.copy2(srt_files[0].name, original_srt_path)

                                # 导出新字幕到输出目录（最终输出文件）
                                new_srt_path = os.path.join(output_dir, f"{project_name}_final.srt")
                                subtitles_state.export(fp=new_srt_path, open_explorer=False)

                                # 步骤2: 获取无声视频路径
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
                                speed_result = adjust_video_speed_by_subtitles(
                                    video_path=silent_video_path,
                                    original_srt_path=original_srt_path,
                                    new_srt_path=new_srt_path,
                                    output_dir=temp_dir,
                                    max_workers=4,
                                    use_gpu=True
                                )

                                if not speed_result['success']:
                                    return gr.update(value=f"❌ **视频变速处理失败**\n\n🎬 **错误**: {speed_result['message']}")

                                speed_adjusted_video = speed_result['output_path']

                                # 步骤4: 获取生成的音频文件路径
                                audio_file_path = os.path.join(current_path, "SAVAdata", "output", f"{subtitles_state.dir}.wav")

                                if not os.path.exists(audio_file_path):
                                    return gr.update(value="❌ **音频文件不存在**\n\n🎵 **错误**: 找不到生成的音频文件")

                                # 步骤5: 合成变速视频与音频
                                from Sava_Utils.video_speed_adjuster import merge_video_with_audio

                                output_video_path = os.path.join(current_path, "SAVAdata", "output", f"{subtitles_state.dir}_final.mp4")

                                final_video = merge_video_with_audio(
                                    video_path=speed_adjusted_video,
                                    audio_path=audio_file_path,
                                    output_path=output_video_path,
                                    use_gpu=True,
                                    sync_to_audio=True
                                )

                                # 生成成功信息
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
• 音频成功率: {success_count/total_count*100:.1f}%

📁 **输出文件**
• 🎬 最终视频: `{final_video}`
• 📂 保存位置: `{os.path.dirname(final_video)}`

🎉 **合成成功！**
您的视频已经成功合成，包含了同步的音频和调整后的播放速度。

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


                        # 绑定合成视频事件
                        compose_video_btn.click(
                            handle_compose_video,
                            inputs=[local_video_path_input, input_file, processing_state, STATE, audio_output],
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
                                    bv2regenbtn = gr.Button(value="🔄️", scale=1, min_width=50, visible=False)
                                    bv2regenbtn.click(remake,
                                                      inputs=[page_slider, edit_real_index, edit_start_end_time, s_txt,
                                                              *BV2_ARGS, STATE],
                                                      outputs=[audio_player, page_slider] + edit_rows[-6:])
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
                                    edit_rows.append(bv2regenbtn)
                                    edit_rows.append(gsvregenbtn)
                                    edit_rows.append(edgettsregenbtn)
                                    edit_rows.append(indexttsregenbtn)
                                    edit_rows.append(customregenbtn)
                        workrefbtn.click(getworklist, inputs=[], outputs=[worklist])
                        export_btn.click(export_subtitle_with_new_name, inputs=[input_file, STATE], outputs=[input_file])
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
                            all_regen_btn_bv2 = gr.Button(value=i18n('Continue Generation'), variant="primary",
                                                          visible=False, interactive=True, min_width=50)
                            edit_rows.append(all_regen_btn_bv2)
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
                            all_regen_btn_bv2.click(
                                lambda process=gr.Progress(track_tqdm=True), *args: gen_multispeaker(*args,
                                                                                                     remake=True),
                                inputs=[INTERRUPT_EVENT, page_slider, workers, *BV2_ARGS, STATE], outputs=edit_rows)
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
                        recompose_btn.click(recompose, inputs=[page_slider, STATE],
                                            outputs=[audio_output, gen_textbox_output_text, *edit_rows])

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
                        select_spk_projet = gr.Dropdown(choices=['bv2', 'gsv', 'edgetts', 'indextts', 'custom'],
                                                        value='gsv', interactive=True, label=i18n('TTS Project'))
                        refresh_spk_list_btn = gr.Button(value="🔄️", min_width=60, scale=0)
                        refresh_spk_list_btn.click(getspklist, inputs=[], outputs=[speaker_list])
                        apply_btn = gr.Button(value="✅", min_width=60, scale=0)
                        apply_btn.click(apply_spk, inputs=[speaker_list, page_slider, STATE, *edit_check_list,
                                                           *edit_real_index_list],
                                        outputs=[*edit_check_list, *edit_rows])

                        save_spk_btn_bv2 = gr.Button(value="💾", min_width=60, scale=0, visible=False)
                        save_spk_btn_bv2.click(lambda *args: save_spk(*args, project="bv2"),
                                               inputs=[speaker_list, *BV2_ARGS], outputs=[speaker_list])
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
                                                 outputs=[save_spk_btn_bv2, save_spk_btn_gsv, save_spk_btn_edgetts,
                                                          save_spk_btn_indextts, save_spk_btn_custom])

                        del_spk_list_btn = gr.Button(value="🗑️", min_width=60, scale=0)
                        del_spk_list_btn.click(del_spk, inputs=[speaker_list], outputs=[speaker_list])
                        start_gen_multispeaker_btn = gr.Button(value=i18n('Start Multi-speaker Synthesizing'),
                                                               variant="primary")
                        start_gen_multispeaker_btn.click(
                            lambda process=gr.Progress(track_tqdm=True), *args: gen_multispeaker(*args),
                            inputs=[INTERRUPT_EVENT, page_slider, workers, STATE], outputs=edit_rows + [audio_output])
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
        BV2.gen_btn1.click(
            lambda process=gr.Progress(track_tqdm=True), *args: generate_preprocess(*args, project="bv2"),
            inputs=[INTERRUPT_EVENT, input_file, fps, offset, workers, *BV2_ARGS],
            outputs=[audio_output, gen_textbox_output_text, worklist, page_slider, *edit_rows, STATE])
        GSV.gen_btn2.click(
            lambda process=gr.Progress(track_tqdm=True), *args: generate_preprocess(*args, project="gsv"),
            inputs=[INTERRUPT_EVENT, input_file, fps, offset, workers, *GSV_ARGS],
            outputs=[audio_output, gen_textbox_output_text, worklist, page_slider, *edit_rows, STATE])
        EDGETTS.gen_btn_edge.click(
            lambda process=gr.Progress(track_tqdm=True), *args: generate_preprocess(*args, project="edgetts"),
            inputs=[INTERRUPT_EVENT, input_file, fps, offset, workers, *EDGETTS_ARGS],
            outputs=[audio_output, gen_textbox_output_text, worklist, page_slider, *edit_rows, STATE])
        INDEXTTS.gen_btn5.click(
            lambda process=gr.Progress(track_tqdm=True), *args: generate_preprocess(*args, project="indextts"),
            inputs=[INTERRUPT_EVENT, input_file, fps, offset, workers, *INDEXTTS_ARGS],
            outputs=[audio_output, gen_textbox_output_text, worklist, page_slider, *edit_rows, STATE])
        CUSTOM.gen_btn4.click(
            lambda process=gr.Progress(track_tqdm=True), *args: generate_preprocess(*args, project="custom"),
            inputs=[INTERRUPT_EVENT, input_file, fps, offset, workers, CUSTOM.choose_custom_api],
            outputs=[audio_output, gen_textbox_output_text, worklist, page_slider, *edit_rows, STATE])
        # Stability is not ensured due to the mechanism of gradio.

    app.queue(default_concurrency_limit=Sava_Utils.config.concurrency_count,
              max_size=2 * Sava_Utils.config.concurrency_count).launch(
        share=args.share,
        server_port=server_port if server_port > 0 else None,
        inbrowser=True,
        server_name='0.0.0.0' if Sava_Utils.config.LAN_access or args.LAN_access else '127.0.0.1',
        show_api=not Sava_Utils.config.server_mode,
    )
