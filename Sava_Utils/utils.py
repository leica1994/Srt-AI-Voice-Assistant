# 所有import语句统一放在文件顶部
import os
import time
import subprocess
import tempfile
import shutil
import platform
import csv
import re
import gradio as gr
import numpy as np

# 项目内部导入
from . import logger, i18n
from .librosa_load import get_rms
from .subtitle_processor import format_ass_file, extract_ass_to_srt, get_available_styles, convert_subtitle
from .subtitle import Subtitles, Subtitle
from .edit_panel import getworklist, load_page
import Sava_Utils


current_path = os.environ.get("current_path")
system = platform.system()
LABELED_TXT_PATTERN = re.compile(r'^([^:：]{1,20})[:：](.+)')


class Flag:
    def __init__(self):
        self.stop = False
        self.using = False

    def set(self):
        if self.using:
            self.stop = True
            return i18n('After completing the generation of the next item, the task will be aborted.')
        else:
            return i18n('No running tasks.')

    def clear(self):
        self.stop = False
        self.using = False

    def is_set(self):
        return self.stop

    def __enter__(self):
        self.stop = False
        self.using = True
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.clear()


def positive_int(*a):
    r = [max(0,int(x)) for x in a]
    return r if len(r)>1 else r[0]

NULL = [None, '', 'None']
def fix_null(*a):
    r = [None if x in NULL else x for x in a]
    return r if len(r) > 1 else r[0]

def basename_no_ext(path: str):
    return os.path.basename(os.path.splitext(path)[0])


def clear_cache():
    dir = os.path.join(current_path, "SAVAdata", "temp")
    if os.path.exists(dir):
        shutil.rmtree(dir)
        logger.info(i18n('Temporary files cleared successfully!'))
        gr.Info(i18n('Temporary files cleared successfully!'))
    else:
        logger.info(i18n('There are no temporary files.'))
        gr.Info(i18n('There are no temporary files.'))


def rc_open_window(command, dir=current_path):
    if system != "Windows":
        subprocess.Popen(command, cwd=dir, shell=True)
        return
    command = f'start cmd /k "{command}"'
    subprocess.Popen(command, cwd=dir, shell=True)
    logger.info(f"{i18n('Execute command')}:{command}")
    time.sleep(0.1)


def rc_bg(command, dir=current_path):
    process = subprocess.Popen(command, cwd=dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, text=True)
    logger.info(f"{i18n('Execute command')}:{command}")
    return process


def kill_process(pid):
    if pid < 0:
        gr.Info(i18n('No running processes'))
        return None
    if system == "Windows":
        command = f"taskkill /t /f /pid {pid}"
    else:
        command = f"pkill --parent {pid} && kill {pid} "  # not tested on real machine yet!!!
    subprocess.run(command, shell=True)
    logger.info(f"{i18n('Execute command')}:{command}")
    gr.Info(i18n('Process terminated.'))


def file_show(files):
    if files in [None, []]:
        return ""
    if len(files) > 1:
        return i18n('<Multiple Files>')
    else:
        file = files[0]

    try:
        file_ext = file.name[-4:].lower()

        # 处理 ASS 文件
        if file_ext == ".ass":
            # 创建临时目录
            temp_dir = tempfile.mkdtemp()
            formatted_ass_path = os.path.join(temp_dir, "formatted.ass")
            srt_output_path = os.path.join(temp_dir, "converted.srt")

            try:
                # 1. 格式化 ASS 文件
                format_success = format_ass_file(file.name, formatted_ass_path)
                if not format_success:
                    logger.warning("ASS 文件格式化失败，使用原文件")
                    formatted_ass_path = file.name

                # 2. 获取可用的样式
                styles = get_available_styles(formatted_ass_path)
                if not styles:
                    logger.warning("未找到 ASS 样式，使用默认样式")
                    style_name = "Default"
                else:
                    style_name = styles[0]  # 使用第一个样式

                # 3. 转换为 SRT
                convert_success = extract_ass_to_srt(formatted_ass_path, style_name, srt_output_path)
                if convert_success and os.path.exists(srt_output_path):
                    with open(srt_output_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    return text
                else:
                    logger.error("ASS 转 SRT 失败")
                    return "❌ ASS 文件处理失败"

            finally:
                # 清理临时文件
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass

        # 处理 VTT 文件
        elif file_ext == ".vtt":
            # 创建临时 SRT 文件
            temp_dir = tempfile.mkdtemp()
            srt_output_path = os.path.join(temp_dir, "converted.srt")

            try:
                # 转换 VTT 为 SRT
                convert_success = convert_subtitle(file.name, srt_output_path)
                if convert_success and os.path.exists(srt_output_path):
                    with open(srt_output_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    return text
                else:
                    logger.error("VTT 转 SRT 失败")
                    return "❌ VTT 文件处理失败"

            finally:
                # 清理临时文件
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass

        # 处理其他格式文件
        else:
            with open(file.name, "r", encoding="utf-8") as f:
                text = f.read()
            return text

    except Exception as error:
        logger.error(f"文件显示失败: {error}")
        return str(error)


from .subtitle import Base_subtitle, Subtitle, Subtitles
from .edit_panel import *


def read_srt(filename, offset):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            file = f.readlines()
        subtitle_list = Subtitles()
        indexlist = []
        filelength = len(file)
        pattern = re.compile(r"\d+")
        for i in range(0, filelength):
            if " --> " in file[i]:
                if pattern.fullmatch(file[i - 1].strip().replace("\ufeff", "")):
                    indexlist.append(i)  # get line id
        listlength = len(indexlist)
        id = 1
        for i in range(0, listlength - 1):
            st, et = file[indexlist[i]].split(" --> ")
            # id = int(file[indexlist[i] - 1].strip().replace("\ufeff", ""))
            text = "".join(file[x] for x in range(indexlist[i] + 1, indexlist[i + 1] - 2))
            st = Subtitle(id, st, et, text, ntype="srt")
            st.add_offset(offset=offset)
            subtitle_list.append(st)
            id += 1
        st, et = file[indexlist[-1]].split(" --> ")
        # id = int(file[indexlist[-1] - 1].strip().replace("\ufeff", ""))
        text = "".join(file[x] for x in range(indexlist[-1] + 1, filelength))
        st = Subtitle(id, st, et, text, ntype="srt")
        st.add_offset(offset=offset)
        subtitle_list.append(st)
    except Exception as e:
        err = f"{i18n('Failed to read file')}: {str(e)}"
        logger.error(err)
        gr.Warning(err)
    return subtitle_list


def read_prcsv(filename, fps, offset):
    try:
        with open(filename, "r", encoding="utf-8", newline="") as csvfile:
            reader = list(csv.reader(csvfile))
            lenth = len(reader)
            subtitle_list = Subtitles()
            stid = 1
            for index in range(1, lenth):
                if reader[index] == []:
                    continue
                st = Subtitle(
                    stid,
                    reader[index][0],
                    reader[index][1],
                    reader[index][2],
                    ntype="prcsv",
                    fps=fps,
                )
                st.add_offset(offset=offset)
                subtitle_list.append(st)
                stid += 1
    except Exception as e:
        err = f"{i18n('Failed to read file')}: {str(e)}"
        logger.error(err)
        gr.Warning(err)
    return subtitle_list


def read_txt(filename):
    # REF_DUR = 2
    try:
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()
        sentences = re.split(r"(?<=[!?。！？])(?=[^!?。！？（）()[\]【】'\"“”]|$)|\n|(?<=[.])(?=\s|$)", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        subtitle_list = Subtitles()
        idx = 1
        for s in sentences:
            subtitle_list.append(Subtitle(idx, "00:00:00,000", "00:00:00,000", s, ntype="srt"))
            idx += 1
    except Exception as e:
        err = f"{i18n('Failed to read file')}: {str(e)}"
        logger.error(err)
        gr.Warning(err)
    return subtitle_list


def read_labeled_txt(filename: str, spk_dict: dict):
    try:
        idx = 1
        subtitle_list = Subtitles()
        subtitle_list.append(Subtitle(idx, "00:00:00,000", "00:00:00,000", "", ntype="srt"))
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("#") or line.strip() == "":
                    continue
                match = LABELED_TXT_PATTERN.match(line.strip())
                if match:
                    speaker = match.group(1).strip()
                    speaker = spk_dict.get(speaker, speaker)
                    if speaker in ['', 'None']:
                        speaker = None
                    subtitle_list.append(Subtitle(idx, "00:00:00,000", "00:00:00,000", match.group(2).strip(), ntype="srt", speaker=speaker))
                    idx += 1
                else:
                    subtitle_list[-1].text += ',' + line
            if not subtitle_list[0].text:
                subtitle_list.pop(0)
        return subtitle_list
    except Exception as e:
        err = f"{i18n('Failed to read file')}: {str(e)}"
        logger.error(err)
        gr.Warning(err)
    return subtitle_list


def get_speaker_map_from_file(in_files):
    speakers = set()
    if in_files in [[], None] or len(in_files) > 1:
        gr.Info(i18n('Creating a multi-speaker project can only upload one file at a time!'))
        return speakers, dict()
    filename = in_files[0].name
    subtitles = read_labeled_file(filename, spk_dict={}, fps=30, offset=0)
    for i in subtitles:
        if i.speaker:
            speakers.add(i.speaker)
        else:
            speakers.add("None")
    speakers_dict = {i:i for i in speakers}        
    return speakers,speakers_dict

def get_speaker_map_from_sub(subtitles:Subtitles):
    speakers = set()
    if subtitles is None or len(subtitles) == 0:
        gr.Info(i18n('There is no subtitle in the current workspace'))
        return speakers, dict()
    for i in subtitles:
        if i.speaker:
            speakers.add(i.speaker)
        else:
            speakers.add("None")
    speakers_dict = {i: i for i in speakers}
    return speakers, speakers_dict


def modify_spkmap(map: dict, k: str, v: str):
    value = v.strip()
    map[k]= value if value!='None' else None


def read_file(file_name, fps=30, offset=0):
    if Sava_Utils.config.server_mode:
        assert os.stat(file_name).st_size < 65536, i18n('Error: File too large')  # 64KB

    file_ext = file_name[-4:].lower()

    if file_ext == ".csv":
        subtitle_list = read_prcsv(file_name, fps, offset)
    elif file_ext == ".srt":
        subtitle_list = read_srt(file_name, offset)
    elif file_ext == ".txt":
        subtitle_list = read_txt(file_name)
    elif file_ext == ".ass":
        subtitle_list = read_ass_file(file_name, offset)
    elif file_ext == ".vtt":
        subtitle_list = read_vtt_file(file_name, offset)
    else:
        raise ValueError(i18n('Unknown format. Please ensure the extension name is correct!'))

    assert len(subtitle_list) != 0, "Empty file???"
    return subtitle_list


def read_labeled_file(file_name, spk_dict, fps=30, offset=0):
    if file_name[-4:].lower() == ".txt":
        subtitle_list = read_labeled_txt(file_name, spk_dict)
    else:
        try:
            subtitle_list = read_file(file_name, fps, offset)
        except Exception as e:
            gr.Warning(str(e))
            return Subtitles()
        for i in subtitle_list:
            match = LABELED_TXT_PATTERN.match(i.text.strip())
            if match:
                speaker = match.group(1).strip()
                speaker = spk_dict.get(speaker, speaker)
                if speaker in ['', 'None']:
                    speaker = None
                i.speaker = speaker
                i.text = match.group(2).strip()
    return subtitle_list    


def create_multi_speaker(in_files, use_labled_text_mode, spk_dict, fps, offset):
    if in_files in [[], None] or len(in_files) > 1:
        gr.Info(i18n('Creating a multi-speaker project can only upload one file at a time!'))
        return getworklist(), *load_page(Subtitles()), Subtitles()
    in_file = in_files[0]
    try:
        if not use_labled_text_mode:
            subtitle_list = read_file(in_file.name, fps, offset)
        else:
            subtitle_list = read_labeled_file(in_file.name, spk_dict, fps, offset)
            assert len(subtitle_list) != 0, "Empty???"
    except Exception as e:
        what = str(e)
        gr.Warning(what)
        return getworklist(), *load_page(Subtitles()), Subtitles()
    subtitle_list.set_dir_name(os.path.basename(in_file.name).replace(".", "-"))
    return getworklist(value=subtitle_list.dir), *load_page(subtitle_list), subtitle_list


def remove_silence(audio, sr, padding_begin=0.1, padding_fin=0.15, threshold_db=-27):
    # Padding(sec) is actually margin of safety
    hop_length = 512
    rms_list = get_rms(audio, hop_length=hop_length).squeeze(0)
    threshold = 10 ** (threshold_db / 20.0)
    x = rms_list > threshold
    i = np.argmax(x)
    j = rms_list.shape[-1] - 1 - np.argmax(x[::-1])
    if not np.any(x) or i==j:
        return audio
    cutting_point1 = max(i * hop_length - int(padding_begin * sr), 0)
    cutting_point2 = min(j * hop_length + int(padding_fin * sr), audio.shape[-1])
    #print(audio.shape[-1],cutting_point1,cutting_point2)
    return audio[cutting_point1:cutting_point2]


def read_ass_file(file_name, offset=0):
    """读取 ASS 文件并转换为字幕列表"""
    try:
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        formatted_ass_path = os.path.join(temp_dir, "formatted.ass")
        srt_output_path = os.path.join(temp_dir, "converted.srt")

        try:
            # 1. 格式化 ASS 文件
            format_success = format_ass_file(file_name, formatted_ass_path)
            if not format_success:
                logger.warning("ASS 文件格式化失败，使用原文件")
                formatted_ass_path = file_name

            # 2. 获取可用的样式
            styles = get_available_styles(formatted_ass_path)
            if not styles:
                logger.warning("未找到 ASS 样式，使用默认样式")
                style_name = "Default"
            else:
                style_name = styles[0]  # 使用第一个样式

            # 3. 转换为 SRT
            convert_success = extract_ass_to_srt(formatted_ass_path, style_name, srt_output_path)
            if convert_success and os.path.exists(srt_output_path):
                # 读取转换后的 SRT 文件
                subtitle_list = read_srt(srt_output_path, offset)
                return subtitle_list
            else:
                logger.error("ASS 转 SRT 失败")
                raise ValueError("ASS 文件处理失败")

        finally:
            # 清理临时文件
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

    except Exception as e:
        err = f"{i18n('Failed to read ASS file')}: {str(e)}"
        logger.error(err)
        raise ValueError(err)


def read_vtt_file(file_name, offset=0):
    """读取 VTT 文件并转换为字幕列表"""
    try:
        # 创建临时 SRT 文件
        temp_dir = tempfile.mkdtemp()
        srt_output_path = os.path.join(temp_dir, "converted.srt")

        try:
            # 转换 VTT 为 SRT
            convert_success = convert_subtitle(file_name, srt_output_path)
            if convert_success and os.path.exists(srt_output_path):
                # 读取转换后的 SRT 文件
                subtitle_list = read_srt(srt_output_path, offset)
                return subtitle_list
            else:
                logger.error("VTT 转 SRT 失败")
                raise ValueError("VTT 文件处理失败")

        finally:
            # 清理临时文件
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

    except Exception as e:
        err = f"{i18n('Failed to read VTT file')}: {str(e)}"
        logger.error(err)
        raise ValueError(err)
