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
    """Demucs 分离器包装类 - 优化显存使用"""

    def __init__(self, model_name: str = "htdemucs", max_memory_gb: float = 8.0):
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_memory_gb = max_memory_gb
        self.chunk_size = self._calculate_chunk_size()
        print(f"🤖 Initializing Demucs model: {model_name} on {self.device}")
        print(f"💾 Memory limit: {max_memory_gb}GB, Chunk size: {self.chunk_size}s")

    def _calculate_chunk_size(self):
        """根据可用显存计算合适的音频块大小"""
        if self.device == "cpu":
            return 60  # CPU模式使用较大块

        try:
            # 获取GPU显存信息
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            available_memory = min(total_memory * 0.8, self.max_memory_gb)  # 使用80%显存或用户限制

            # 根据可用显存计算块大小（经验公式）
            if available_memory >= 12:
                chunk_size = 60  # 12GB+: 60秒块
            elif available_memory >= 8:
                chunk_size = 45  # 8-12GB: 45秒块
            elif available_memory >= 6:
                chunk_size = 30  # 6-8GB: 30秒块
            elif available_memory >= 4:
                chunk_size = 20  # 4-6GB: 20秒块
            else:
                chunk_size = 15  # <4GB: 15秒块

            print(f"🔍 GPU Memory: {total_memory:.1f}GB total, using {available_memory:.1f}GB")
            return chunk_size

        except Exception as e:
            print(f"⚠️ Could not detect GPU memory: {e}, using conservative chunk size")
            return 20  # 保守的块大小

    def _load_model(self):
        """加载 Demucs 模型"""
        if self.model is None:
            print(f"📥 Loading Demucs model: {self.model_name}...")
            try:
                self.model = get_model(self.model_name)
                self.model.to(self.device)
                self.model.eval()

                # 设置模型为半精度以节省显存
                if self.device == "cuda":
                    self.model = self.model.half()

                print("✅ Model loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to load Demucs model: {e}")

    def separate_audio_file(self, audio_path: str):
        """使用 Python API 分离音频文件 - 优化显存使用"""
        self._load_model()

        print(f"🎵 Using Demucs Python API to separate audio...")

        try:
            # 加载音频信息
            info = torchaudio.info(audio_path)
            sample_rate = info.sample_rate
            total_frames = info.num_frames
            duration = total_frames / sample_rate

            print(f"📊 Audio info: {duration:.1f}s, {sample_rate}Hz, {info.num_channels} channels")

            # 检查是否需要分块处理
            if duration <= self.chunk_size:
                return self._process_single_chunk(audio_path)
            else:
                return self._process_with_chunks(audio_path, sample_rate, total_frames)

        except Exception as e:
            raise RuntimeError(f"Demucs separation failed: {e}")
        finally:
            # 清理 GPU 内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

    def _process_single_chunk(self, audio_path: str):
        """处理单个音频块"""
        print("🔄 Processing as single chunk...")

        # 加载音频
        waveform, sample_rate = torchaudio.load(audio_path)

        # 确保音频是立体声
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]

        # 转换为半精度以节省显存
        if self.device == "cuda":
            waveform = waveform.half()

        # 移动到设备
        waveform = waveform.to(self.device)

        # 应用模型进行分离
        with torch.no_grad():
            sources = apply_model(self.model, waveform.unsqueeze(0),
                                  device=self.device, progress=True)[0]

        return self._save_sources(sources, sample_rate, Path(audio_path).stem)

    def _process_with_chunks(self, audio_path: str, sample_rate: int, total_frames: int):
        """分块处理大音频文件"""
        chunk_frames = int(self.chunk_size * sample_rate)
        overlap_frames = int(0.5 * sample_rate)  # 0.5秒重叠

        num_chunks = (total_frames + chunk_frames - 1) // chunk_frames
        print(f"🧩 Processing in {num_chunks} chunks of {self.chunk_size}s each...")

        # 初始化输出容器
        source_names = self.model.sources
        accumulated_sources = {name: [] for name in source_names}

        for chunk_idx in range(num_chunks):
            start_frame = chunk_idx * chunk_frames
            end_frame = min(start_frame + chunk_frames + overlap_frames, total_frames)

            print(f"📦 Processing chunk {chunk_idx + 1}/{num_chunks} "
                  f"({start_frame/sample_rate:.1f}s - {end_frame/sample_rate:.1f}s)")

            try:
                # 加载音频块
                waveform, _ = torchaudio.load(
                    audio_path,
                    frame_offset=start_frame,
                    num_frames=end_frame - start_frame
                )

                # 确保立体声
                if waveform.shape[0] == 1:
                    waveform = waveform.repeat(2, 1)
                elif waveform.shape[0] > 2:
                    waveform = waveform[:2]

                # 转换为半精度
                if self.device == "cuda":
                    waveform = waveform.half()

                waveform = waveform.to(self.device)

                # 分离音频
                with torch.no_grad():
                    sources = apply_model(self.model, waveform.unsqueeze(0),
                                          device=self.device, progress=False)[0]

                # 处理重叠部分
                if chunk_idx > 0:
                    # 移除前半秒重叠
                    fade_frames = int(0.25 * sample_rate)  # 0.25秒淡入淡出
                    sources = sources[:, :, fade_frames:]

                if chunk_idx < num_chunks - 1:
                    # 移除后半秒重叠
                    fade_frames = int(0.25 * sample_rate)
                    sources = sources[:, :, :-fade_frames]

                # 移到CPU并累积
                for i, source_name in enumerate(source_names):
                    source_chunk = sources[i].cpu().float()
                    accumulated_sources[source_name].append(source_chunk)

                # 清理当前块的GPU内存
                del sources, waveform
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"⚠️ Error processing chunk {chunk_idx + 1}: {e}")
                continue

        # 合并所有块
        print("🔗 Merging chunks...")
        final_sources = {}
        for i, source_name in enumerate(source_names):
            if accumulated_sources[source_name]:
                merged = torch.cat(accumulated_sources[source_name], dim=1)
                final_sources[source_name] = merged

        return self._save_sources_dict(final_sources, sample_rate, Path(audio_path).stem)

    def _save_sources(self, sources, sample_rate: int, audio_name: str):
        """保存分离的音频源"""
        outputs = {}
        source_names = self.model.sources

        for i, source_name in enumerate(source_names):
            # 移回 CPU 并转换
            source_audio = sources[i].cpu().float()

            # 创建临时文件
            temp_file = Path(tempfile.gettempdir()) / f"demucs_{source_name}_{audio_name}.wav"

            # 保存音频
            torchaudio.save(str(temp_file), source_audio, sample_rate)
            outputs[source_name] = str(temp_file)

        return outputs

    def _save_sources_dict(self, sources_dict, sample_rate: int, audio_name: str):
        """保存分离的音频源字典"""
        outputs = {}

        for source_name, source_audio in sources_dict.items():
            # 创建临时文件
            temp_file = Path(tempfile.gettempdir()) / f"demucs_{source_name}_{audio_name}.wav"

            # 保存音频
            torchaudio.save(str(temp_file), source_audio, sample_rate)
            outputs[source_name] = str(temp_file)

        return outputs


class AudioSeparator:
    """音频分离器主类"""

    QUALITY_SETTINGS = {
        'low': {'bitrate': '64k', 'samplerate': '22050', 'description': '低质量 - 快速处理'},
        'medium': {'bitrate': '128k', 'samplerate': '44100', 'description': '中等质量 - 平衡'},
        'high': {'bitrate': '256k', 'samplerate': '48000', 'description': '高质量 - 最佳效果'},
        'ultra_low': {'bitrate': '32k', 'samplerate': '16000', 'description': '超低质量 - 超大文件专用'}
    }

    def __init__(self, output_dir: str = "output", model_name: str = "htdemucs", max_memory_gb: float = 8.0, auto_quality: bool = True):
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.max_memory_gb = max_memory_gb
        self.auto_quality = auto_quality
        self.separator = None
        self.temp_files = []

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_video_info(self, video_path: str) -> dict:
        """获取视频文件信息"""
        try:
            probe = ffmpeg.probe(video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            audio_info = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)

            duration = float(probe['format']['duration'])
            file_size = int(probe['format']['size']) / (1024 * 1024)  # MB

            return {
                'duration': duration,
                'file_size_mb': file_size,
                'video_codec': video_info.get('codec_name', 'unknown'),
                'audio_codec': audio_info.get('codec_name', 'unknown') if audio_info else 'no_audio',
                'width': int(video_info.get('width', 0)),
                'height': int(video_info.get('height', 0))
            }
        except Exception as e:
            print(f"⚠️ Could not get video info: {e}")
            return {'duration': 0, 'file_size_mb': 0}

    def _smart_quality_selection(self, video_path: str, requested_quality: str) -> str:
        """智能质量选择 - 根据文件大小和时长自动调整"""
        if not self.auto_quality:
            return requested_quality

        info = self._get_video_info(video_path)
        duration_hours = info['duration'] / 3600
        file_size_mb = info['file_size_mb']

        print(f"📊 Video analysis: {duration_hours:.1f}h, {file_size_mb:.1f}MB")

        # 智能质量选择规则
        if duration_hours > 2.0 or file_size_mb > 2000:  # 超过2小时或2GB
            recommended_quality = 'ultra_low'
            reason = f"超大文件 ({duration_hours:.1f}h, {file_size_mb:.0f}MB)"
        elif duration_hours > 1.0 or file_size_mb > 1000:  # 超过1小时或1GB
            recommended_quality = 'low'
            reason = f"大文件 ({duration_hours:.1f}h, {file_size_mb:.0f}MB)"
        elif duration_hours > 0.5 or file_size_mb > 500:  # 超过30分钟或500MB
            recommended_quality = 'medium'
            reason = f"中等文件 ({duration_hours:.1f}h, {file_size_mb:.0f}MB)"
        else:
            recommended_quality = requested_quality  # 保持用户选择
            reason = f"小文件 ({duration_hours:.1f}h, {file_size_mb:.0f}MB)"

        # 如果推荐质量低于用户请求，给出提示
        quality_levels = {'ultra_low': 0, 'low': 1, 'medium': 2, 'high': 3}
        if quality_levels.get(recommended_quality, 2) < quality_levels.get(requested_quality, 2):
            print(f"🎯 智能质量调整: {requested_quality} → {recommended_quality}")
            print(f"   原因: {reason}")
            print(f"   说明: {self.QUALITY_SETTINGS[recommended_quality]['description']}")
            print(f"   💡 这将显著减少处理时间和显存使用")
            return recommended_quality
        else:
            print(f"✅ 保持用户选择的质量: {requested_quality} ({reason})")
            return requested_quality

    def _estimate_processing_time(self, duration_hours: float, quality: str) -> str:
        """估算处理时间"""
        # 基于经验的处理时间估算（分钟）
        base_time = {
            'ultra_low': duration_hours * 5,   # 5分钟/小时
            'low': duration_hours * 8,         # 8分钟/小时
            'medium': duration_hours * 15,     # 15分钟/小时
            'high': duration_hours * 25        # 25分钟/小时
        }

        estimated_minutes = base_time.get(quality, duration_hours * 15)

        if estimated_minutes < 1:
            return "< 1分钟"
        elif estimated_minutes < 60:
            return f"约 {estimated_minutes:.0f}分钟"
        else:
            hours = estimated_minutes / 60
            return f"约 {hours:.1f}小时"

    def extract_audio_from_video(self, video_path: str, audio_path: str = None,
                                 audio_quality: str = "high") -> str:
        """从视频中提取音频 - 支持智能质量调整"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # 智能质量选择
        final_quality = self._smart_quality_selection(video_path, audio_quality)

        if audio_path is None:
            video_name = Path(video_path).stem
            audio_path = self.output_dir / f"{video_name}_raw.wav"

        settings = self.QUALITY_SETTINGS[final_quality]

        # 获取视频信息用于时间估算
        info = self._get_video_info(video_path)
        duration_hours = info['duration'] / 3600
        estimated_time = self._estimate_processing_time(duration_hours, final_quality)

        print(f"🎬 提取音频: {final_quality} 质量 ({settings['description']})")
        print(f"⏱️ 预计处理时间: {estimated_time}")

        if final_quality != audio_quality:
            print(f"🔄 质量已自动调整: {audio_quality} → {final_quality} (优化大文件处理)")

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
            print(f"✅ 音频提取完成: {audio_path}")
            print(f"📊 输出参数: {settings['bitrate']} @ {settings['samplerate']}Hz")
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
                self.separator = DemucsWrapper(self.model_name, self.max_memory_gb)
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
                         audio_quality: str = "high", model_name: str = "htdemucs",
                         max_memory_gb: float = 8.0, auto_quality: bool = True) -> Dict[str, str]:
    """
    从视频分离音频和人声轨道

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        audio_quality: 音频质量 (会根据文件大小自动调整)
        model_name: Demucs 模型名称
        max_memory_gb: 最大显存使用限制(GB)
        auto_quality: 是否启用智能质量调整

    Returns:
        Dict[str, str]: 包含所有输出文件路径的字典
    """
    with AudioSeparator(output_dir=output_dir, model_name=model_name,
                       max_memory_gb=max_memory_gb, auto_quality=auto_quality) as separator:
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

        segment_filename = f"segment_{subtitle['number']:06d}.wav"
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