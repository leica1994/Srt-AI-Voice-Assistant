"""
优化的音频分离模块
"""

import os
import gc
import re
import datetime
import warnings
from typing import Tuple, Dict, List
from pathlib import Path
import tempfile
import shutil

# 隐藏 PyTorch 和 CUDA 相关警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message=".*cudnn.*")
warnings.filterwarnings("ignore", message=".*flash attention.*")
warnings.filterwarnings("ignore", message=".*Plan failed.*")

import ffmpeg
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torch
import torchaudio
from pydub import AudioSegment

# 设置 PyTorch 优化选项
if torch.cuda.is_available():
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


class DemucsWrapper:
    """Demucs 分离器包装类"""

    def __init__(self, model_name: str = "htdemucs"):
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🤖 Initializing Demucs model: {model_name} on {self.device}")

    def _load_model(self):
        """加载 Demucs 模型"""
        if self.model is None:
            print(f"📥 Loading Demucs model: {self.model_name}...")
            try:
                self.model = get_model(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                print("✅ Model loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to load Demucs model: {e}")

    def separate_audio_file(self, audio_path: str):
        """使用 Python API 分离音频文件"""
        self._load_model()

        print(f"🎵 Using Demucs Python API to separate audio...")

        try:
            # 加载音频
            waveform, sample_rate = torchaudio.load(audio_path)

            # 确保音频是立体声
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            elif waveform.shape[0] > 2:
                waveform = waveform[:2]

            # 移动到设备
            waveform = waveform.to(self.device)

            # 应用模型进行分离
            with torch.no_grad():
                sources = apply_model(self.model, waveform.unsqueeze(0),
                                      device=self.device, progress=True)[0]

            # 获取源名称
            source_names = self.model.sources

            # 保存分离的音频
            outputs = {}
            audio_name = Path(audio_path).stem

            for i, source_name in enumerate(source_names):
                # 移回 CPU 并转换
                source_audio = sources[i].cpu()

                # 创建临时文件
                temp_file = Path(tempfile.gettempdir()) / f"demucs_{source_name}_{audio_name}.wav"

                # 保存音频
                torchaudio.save(str(temp_file), source_audio, sample_rate)
                outputs[source_name] = str(temp_file)

            return outputs

        except Exception as e:
            raise RuntimeError(f"Demucs separation failed: {e}")
        finally:
            # 清理 GPU 内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class AudioSeparator:
    """音频分离器主类"""

    QUALITY_SETTINGS = {
        'low': {'bitrate': '64k', 'samplerate': '22050'},
        'medium': {'bitrate': '128k', 'samplerate': '44100'},
        'high': {'bitrate': '256k', 'samplerate': '48000'}
    }

    def __init__(self, output_dir: str = "output", model_name: str = "htdemucs"):
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.separator = None
        self.temp_files = []

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_audio_from_video(self, video_path: str, audio_path: str = None,
                                 audio_quality: str = "high") -> str:
        """从视频中提取音频"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if audio_path is None:
            video_name = Path(video_path).stem
            audio_path = self.output_dir / f"{video_name}_raw.wav"

        settings = self.QUALITY_SETTINGS[audio_quality]
        print(f"🎬 Extracting audio: {audio_quality} quality")

        try:
            stream = ffmpeg.input(str(video_path))
            stream = ffmpeg.output(
                stream,
                str(audio_path),
                vn=None,
                acodec='libmp3lame',
                audio_bitrate=settings['bitrate'],
                ar=settings['samplerate'],
                ac=2
            )
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            print(f"✅ Audio extracted: {audio_path}")
            return str(audio_path)

        except Exception as e:
            raise RuntimeError(f"FFmpeg failed: {e}")

    def extract_video_from_video(self, video_path: str, video_path_output: str = None) -> str:
        """从视频中提取无声视频（移除音频轨道）"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if video_path_output is None:
            video_name = Path(video_path).stem
            video_ext = Path(video_path).suffix  # 获取原视频的扩展名
            video_path_output = self.output_dir / f"{video_name}_silent{video_ext}"

        print(f"🎬 Extracting silent video...")

        try:
            stream = ffmpeg.input(str(video_path))
            stream = ffmpeg.output(
                stream,
                str(video_path_output),
                an=None,  # 移除音频
                vcodec='copy'  # 复制视频流，不重新编码
            )
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            print(f"✅ Silent video extracted: {video_path_output}")
            return str(video_path_output)

        except Exception as e:
            raise RuntimeError(f"FFmpeg failed to extract silent video: {e}")

    def _load_demucs_model(self) -> None:
        """加载 Demucs 模型"""
        if self.separator is None:
            try:
                self.separator = DemucsWrapper(self.model_name)
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {e}")

    def separate_audio(self, audio_path: str, output_vocal: str = None,
                       output_background: str = None) -> Tuple[str, str]:
        """分离音频源"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._load_demucs_model()
        output_vocal, output_background = self._prepare_output_paths(
            audio_path, output_vocal, output_background
        )

        print("🎵 Separating audio sources...")

        try:
            outputs = self.separator.separate_audio_file(audio_path)
            self._process_separation_outputs(outputs, output_vocal, output_background)
            print("✨ Audio separation completed!")
            return str(output_vocal), str(output_background)

        except Exception as e:
            raise RuntimeError(f"Audio separation failed: {e}")
        finally:
            gc.collect()

    def _prepare_output_paths(self, audio_path: str, output_vocal: str = None,
                              output_background: str = None) -> Tuple[Path, Path]:
        """准备输出路径"""
        audio_name = Path(audio_path).stem

        if output_vocal is None:
            output_vocal = self.output_dir / f"{audio_name}_vocal.wav"
        else:
            output_vocal = Path(output_vocal)

        if output_background is None:
            output_background = self.output_dir / f"{audio_name}_background.wav"
        else:
            output_background = Path(output_background)

        return output_vocal, output_background

    def _process_separation_outputs(self, outputs: dict, output_vocal: Path,
                                    output_background: Path) -> None:
        """处理音频分离的输出文件"""
        try:
            if 'vocals' not in outputs:
                raise RuntimeError("Vocals track not found")

            shutil.copy2(outputs['vocals'], str(output_vocal))
            print(f"✅ Vocals: {output_vocal}")

            background_sources = [path for name, path in outputs.items() if name != 'vocals']
            if background_sources:
                shutil.copy2(background_sources[0], str(output_background))
                print(f"✅ Background: {output_background}")
            else:
                print("⚠️ No background tracks found")

        finally:
            self._cleanup_temp_files(outputs)

    def _cleanup_temp_files(self, outputs: dict = None) -> None:
        """清理临时文件"""
        files_to_clean = []

        if outputs:
            files_to_clean.extend(outputs.values())

        files_to_clean.extend(self.temp_files)

        for temp_file in files_to_clean:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"🗑️ Cleaned temp file: {os.path.basename(temp_file)}")
            except Exception as e:
                print(f"⚠️ Failed to clean temp file {temp_file}: {e}")

        self.temp_files.clear()

    def process_video(self, video_path: str, audio_quality: str = "medium") -> Dict[str, str]:
        """完整的视频音频分离流程"""
        print(f"🚀 Processing: {Path(video_path).name}")

        try:
            raw_video = self.extract_video_from_video(video_path)
            raw_audio = self.extract_audio_from_video(video_path, audio_quality=audio_quality)
            normalized_audio = self.normalize_audio_volume(raw_audio)
            vocal_audio, background_audio = self.separate_audio(normalized_audio)

            result = {
                'raw_video': raw_video,
                'raw_audio': raw_audio,
                'vocal_audio': vocal_audio,
                'background_audio': background_audio
            }

            print("🎉 Processing completed!")
            return result

        except Exception as e:
            print(f"❌ Processing failed: {e}")
            raise

    @staticmethod
    def normalize_audio_volume(audio_path: str, target_db: float = -20.0) -> str:
        """
        标准化音频音量

        Args:
            audio_path: 音频文件路径
            target_db: 目标音量 (dB)

        Returns:
            str: 标准化后的音频文件路径

        Raises:
            ImportError: pydub 不可用
            FileNotFoundError: 音频文件不存在
        """
        if AudioSegment is None:
            raise ImportError("pydub is required for audio normalization. "
                              "Install with: pip install pydub")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"🔊 Normalizing audio volume to {target_db}dB...")

        try:
            audio = AudioSegment.from_file(audio_path)
            original_db = audio.dBFS
            change_in_dBFS = target_db - original_db
            normalized_audio = audio.apply_gain(change_in_dBFS)

            # 覆盖原文件
            normalized_audio.export(audio_path, format="wav")
            print(f"✅ Audio normalized from {original_db:.1f}dB to {target_db:.1f}dB")

            return audio_path
        except Exception as ex:
            raise RuntimeError(f"Failed to normalize audio: {ex}")

    def cleanup(self) -> None:
        """清理模型和释放内存"""
        try:
            self._cleanup_temp_files()

            if self.separator is not None:
                del self.separator
                self.separator = None
            gc.collect()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def parse_srt_file(srt_path: str) -> List[Dict]:
    """解析 SRT 字幕文件"""
    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'cp1252']
    content = None

    for encoding in encodings:
        try:
            with open(srt_path, 'r', encoding=encoding) as file:
                content = file.read()
            print(f"✅ SRT file loaded with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        raise RuntimeError(f"Could not decode SRT file with any supported encoding")

    subtitles = []
    for block in content.strip().split('\n\n'):
        lines = [line.strip() for line in block.split('\n') if line.strip()]
        if len(lines) < 3:
            continue

        try:
            number = int(lines[0])
            start_time, end_time = lines[1].split(' --> ')
            start_time = _parse_timestamp(start_time)
            end_time = _parse_timestamp(end_time)
            duration = (end_time - start_time).total_seconds()
            text = ' '.join(lines[2:])
            text = re.sub(r'[（\(][^）\)]*[）\)]', '', text).strip()
            text = text.replace('-', '').strip()

            subtitles.append({
                'number': number,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'text': text
            })
        except (ValueError, IndexError):
            continue

    print(f"✅ Parsed {len(subtitles)} subtitle entries")
    return subtitles


def _parse_timestamp(timestamp_str: str) -> datetime.datetime:
    """解析 SRT 时间戳"""
    timestamp_str = timestamp_str.replace(',', '.')
    return datetime.datetime.strptime(timestamp_str, '%H:%M:%S.%f')


def _timestamp_to_seconds(timestamp: datetime.datetime) -> float:
    """将时间戳转换为秒数"""
    return (timestamp.hour * 3600 + timestamp.minute * 60 +
            timestamp.second + timestamp.microsecond / 1000000)


def separate_video_audio(video_path: str, output_dir: str = "output",
                         audio_quality: str = "high", model_name: str = "htdemucs") -> Dict[str, str]:
    """
    从视频分离音频和人声轨道
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        audio_quality: 音频质量
        model_name: Demucs 模型名称
    
    Returns:
        Dict[str, str]: 包含所有输出文件路径的字典
    """
    with AudioSeparator(output_dir=output_dir, model_name=model_name) as separator:
        return separator.process_video(video_path, audio_quality=audio_quality)


def split_audio_by_subtitles(audio_path: str, srt_path: str, segments_output_dir: str = None) -> List[Dict]:
    """
    根据字幕文件分割音频
    
    Args:
        audio_path: 音频文件路径
        srt_path: SRT 字幕文件路径
        segments_output_dir: 输出目录（可选）
    
    Returns:
        List[Dict]: 分割后的音频片段信息
    """
    segments_output_dir = Path("segments") if segments_output_dir is None else Path(segments_output_dir)

    segments_output_dir.mkdir(exist_ok=True)

    print(f"📝 Parsing subtitle file: {srt_path}")
    subtitles = parse_srt_file(srt_path)

    print(f"🎵 Loading audio file: {audio_path}")
    audio = AudioSegment.from_file(audio_path)
    audio_duration_ms = len(audio)
    audio_duration_s = audio_duration_ms / 1000.0

    print(f"🎵 Audio duration: {audio_duration_s:.2f} seconds")
    segments = []

    print(f"✂️ Splitting audio into {len(subtitles)} segments...")

    for subtitle in subtitles:
        start_ms = int(_timestamp_to_seconds(subtitle['start_time']) * 1000)
        end_ms = int(_timestamp_to_seconds(subtitle['end_time']) * 1000)

        # 直接按字幕时间分割
        segment = audio[start_ms:end_ms]

        segment_filename = f"segment_{subtitle['number']:04d}.wav"
        segment_path = segments_output_dir / segment_filename

        segment.export(str(segment_path), format="wav")

        segments.append({
            'number': subtitle['number'],
            'text': subtitle['text'],
            'duration': subtitle['duration'],
            'file_path': str(segment_path),
            'start_time': start_ms / 1000.0,
            'end_time': end_ms / 1000.0
        })

    print(f"✅ Created {len(segments)} segments in {segments_output_dir}")
    return segments