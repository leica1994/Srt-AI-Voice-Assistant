"""
优化的音频分离模块
"""

import os
import gc
import re
import datetime
import warnings
import psutil
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
    """Demucs 分离器包装类 - 优化显存使用，针对16GB显卡"""

    def __init__(self, model_name: str = "htdemucs", max_memory_gb: float = 8.0):
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_memory_gb = max_memory_gb
        self.chunk_size = self._calculate_chunk_size()
        self.aggressive_cleanup = True  # 启用积极的显存清理

        # 检测是否为16GB显卡以启用高质量模式
        self.high_quality_mode = False
        if torch.cuda.is_available():
            try:
                total_gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                self.high_quality_mode = total_gpu_gb >= 15
            except:
                pass

        print(f"🤖 Initializing Demucs model: {model_name} on {self.device}")
        print(f"💾 Memory limit: {max_memory_gb}GB, Chunk size: {self.chunk_size}s")
        print(f"🧹 Aggressive GPU cleanup: {'Enabled' if self.aggressive_cleanup else 'Disabled'}")
        print(f"🎯 High quality mode: {'Enabled' if self.high_quality_mode else 'Disabled'}")

    def _get_cpu_memory_info(self):
        """获取CPU内存信息"""
        try:
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)
            return total_gb, available_gb
        except Exception as e:
            print(f"⚠️ Could not detect CPU memory: {e}")
            return 16.0, 8.0  # 保守估计

    def _calculate_chunk_size(self):
        """根据可用显存和内存计算合适的音频块大小 - 针对16GB显卡优化"""
        # 获取CPU内存信息
        total_cpu_gb, available_cpu_gb = self._get_cpu_memory_info()

        if self.device == "cpu":
            # CPU模式：根据可用内存调整块大小
            if available_cpu_gb >= 8:
                chunk_size = 45  # 8GB+: 45秒块
            elif available_cpu_gb >= 4:
                chunk_size = 30  # 4-8GB: 30秒块
            elif available_cpu_gb >= 2:
                chunk_size = 20  # 2-4GB: 20秒块
            else:
                chunk_size = 10  # <2GB: 10秒块

            print(f"🔍 CPU Memory: {total_cpu_gb:.1f}GB total, {available_cpu_gb:.1f}GB available")
            print(f"💾 CPU mode chunk size: {chunk_size}s")
            return chunk_size

        try:
            # 获取GPU显存信息
            total_gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB

            # 针对16GB显卡的优化配置
            if total_gpu_gb >= 15:  # 16GB显卡
                # 使用保守的显存限制，确保稳定性
                safe_gpu_gb = min(total_gpu_gb * 0.7, self.max_memory_gb)  # 使用70%显存
                print(f"🎯 16GB GPU detected, using conservative 70% limit: {safe_gpu_gb:.1f}GB")
            else:
                # 其他显卡使用80%
                safe_gpu_gb = min(total_gpu_gb * 0.8, self.max_memory_gb)

            # 同时考虑GPU显存和CPU内存限制
            # 音频处理时，GPU处理但CPU需要存储中间结果
            effective_memory = min(safe_gpu_gb, available_cpu_gb * 0.4)  # CPU内存的40%用于缓存

            # 针对16GB显卡的块大小优化 - 积极清理后可以使用更大块
            if total_gpu_gb >= 15:  # 16GB显卡
                if effective_memory >= 10:
                    chunk_size = 120  # 10GB+: 2分钟块，高质量处理
                elif effective_memory >= 8:
                    chunk_size = 90   # 8-10GB: 1.5分钟块
                elif effective_memory >= 6:
                    chunk_size = 75   # 6-8GB: 1.25分钟块
                elif effective_memory >= 4:
                    chunk_size = 60   # 4-6GB: 1分钟块
                else:
                    chunk_size = 45   # <4GB: 45秒块
            else:
                # 其他显卡的配置
                if effective_memory >= 8:
                    chunk_size = 45  # 8GB+: 45秒块
                elif effective_memory >= 6:
                    chunk_size = 30  # 6-8GB: 30秒块
                elif effective_memory >= 4:
                    chunk_size = 20  # 4-6GB: 20秒块
                elif effective_memory >= 2:
                    chunk_size = 15  # 2-4GB: 15秒块
                else:
                    chunk_size = 10  # <2GB: 10秒块

            print(f"🔍 GPU Memory: {total_gpu_gb:.1f}GB total, using {safe_gpu_gb:.1f}GB")
            print(f"🔍 CPU Memory: {total_cpu_gb:.1f}GB total, {available_cpu_gb:.1f}GB available")
            print(f"💾 Effective memory limit: {effective_memory:.1f}GB, chunk size: {chunk_size}s")
            return chunk_size

        except Exception as e:
            print(f"⚠️ Could not detect GPU memory: {e}, using conservative chunk size")
            # 回退到仅基于CPU内存的计算
            if available_cpu_gb >= 4:
                return 20
            elif available_cpu_gb >= 2:
                return 15
            else:
                return 10

    def _aggressive_gpu_cleanup(self):
        """积极的GPU显存清理 - 针对16GB显卡优化"""
        if not torch.cuda.is_available():
            return

        try:
            # 清空CUDA缓存
            torch.cuda.empty_cache()

            # 强制垃圾回收
            gc.collect()

            # 如果启用积极清理，进行更深度的清理
            if self.aggressive_cleanup:
                # 同步CUDA操作
                torch.cuda.synchronize()

                # 再次清空缓存
                torch.cuda.empty_cache()

                # 获取当前显存使用情况
                if hasattr(torch.cuda, 'memory_allocated'):
                    allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                    cached = torch.cuda.memory_reserved() / (1024**3)  # GB
                    print(f"🧹 GPU Memory after cleanup: {allocated:.2f}GB allocated, {cached:.2f}GB cached")

        except Exception as e:
            print(f"⚠️ GPU cleanup warning: {e}")

    def _load_model(self):
        """加载 Demucs 模型"""
        if self.model is None:
            print(f"📥 Loading Demucs model: {self.model_name}...")
            try:
                self.model = get_model(self.model_name)
                self.model.to(self.device)
                self.model.eval()

                # 智能精度优化 - 16GB显卡可选择保持单精度获得更高质量
                self.use_half_precision = False
                if self.device == "cuda":
                    if self.high_quality_mode:
                        # 16GB显卡高质量模式：优先使用单精度
                        print("🎯 16GB显卡高质量模式：使用单精度，确保最佳音质")
                        self.use_half_precision = False
                    else:
                        # 其他情况尝试半精度优化
                        try:
                            # 测试模型是否支持半精度
                            test_input = torch.randn(1, 2, 1000).half().to(self.device)
                            with torch.no_grad():
                                _ = self.model(test_input)

                            # 如果测试成功，启用半精度
                            self.model = self.model.half()
                            self.use_half_precision = True
                            print("✅ 半精度优化已启用，显存使用减少50%")

                        except Exception as e:
                            print(f"⚠️ 半精度不兼容，使用单精度: {e}")
                            self.use_half_precision = False

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
            # 积极清理GPU显存
            self._aggressive_gpu_cleanup()

    def _process_single_chunk(self, audio_path: str):
        """处理单个音频块"""
        print("🔄 Processing as single chunk...")

        try:
            # 加载音频
            waveform, sample_rate = torchaudio.load(audio_path)

            # 确保音频是立体声
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            elif waveform.shape[0] > 2:
                waveform = waveform[:2]

            # 智能精度处理
            waveform = waveform.to(self.device)
            if self.use_half_precision:
                waveform = waveform.half()

            # 应用模型进行分离
            with torch.no_grad():
                sources = apply_model(self.model, waveform.unsqueeze(0),
                                      device=self.device, progress=True)[0]

            # 清理输入数据
            del waveform

            result = self._save_sources(sources, sample_rate, Path(audio_path).stem)

            # 清理分离结果
            del sources

            return result

        finally:
            # 积极清理GPU显存
            self._aggressive_gpu_cleanup()

    def _process_with_chunks(self, audio_path: str, sample_rate: int, total_frames: int):
        """分块处理大音频文件 - 使用流式合并避免内存溢出"""
        chunk_frames = int(self.chunk_size * sample_rate)

        # 16GB显卡高质量模式使用更大的重叠区域
        if self.high_quality_mode:
            overlap_frames = int(1.0 * sample_rate)  # 1秒重叠，提高质量
            print("🎯 16GB显卡高质量模式：使用1秒重叠区域")
        else:
            overlap_frames = int(0.5 * sample_rate)  # 0.5秒重叠

        num_chunks = (total_frames + chunk_frames - 1) // chunk_frames
        print(f"🧩 Processing in {num_chunks} chunks of {self.chunk_size}s each...")

        # 初始化临时文件用于流式合并
        source_names = self.model.sources
        temp_files = {name: [] for name in source_names}

        # 检查可用内存，决定合并策略
        _, available_cpu_gb = self._get_cpu_memory_info()
        use_streaming_merge = available_cpu_gb < 6.0 or num_chunks > 10  # 内存不足或块数太多时使用流式合并

        if use_streaming_merge:
            print(f"💾 Using streaming merge (Available RAM: {available_cpu_gb:.1f}GB)")
        else:
            print(f"💾 Using in-memory merge (Available RAM: {available_cpu_gb:.1f}GB)")
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

                # 智能精度处理
                waveform = waveform.to(self.device)
                if self.use_half_precision:
                    waveform = waveform.half()

                # 分离音频
                with torch.no_grad():
                    sources = apply_model(self.model, waveform.unsqueeze(0),
                                          device=self.device, progress=False)[0]

                # 处理重叠部分 - 高质量模式使用更长的淡入淡出
                if self.high_quality_mode:
                    fade_frames = int(0.5 * sample_rate)  # 0.5秒淡入淡出，更平滑
                else:
                    fade_frames = int(0.25 * sample_rate)  # 0.25秒淡入淡出

                if chunk_idx > 0:
                    # 移除前重叠
                    sources = sources[:, :, fade_frames:]

                if chunk_idx < num_chunks - 1:
                    # 移除后重叠
                    sources = sources[:, :, :-fade_frames]

                # 根据策略处理分离结果
                if use_streaming_merge:
                    # 流式合并：直接保存到临时文件
                    for i, source_name in enumerate(source_names):
                        source_chunk = sources[i].cpu().float()

                        # 创建临时文件
                        temp_file = Path(tempfile.gettempdir()) / f"demucs_chunk_{source_name}_{chunk_idx}.wav"
                        torchaudio.save(str(temp_file), source_chunk, sample_rate)
                        temp_files[source_name].append(str(temp_file))

                        # 立即清理内存
                        del source_chunk
                else:
                    # 内存合并：累积到内存中
                    for i, source_name in enumerate(source_names):
                        source_chunk = sources[i].cpu().float()
                        accumulated_sources[source_name].append(source_chunk)

                # 清理当前块的GPU内存
                del sources, waveform
                self._aggressive_gpu_cleanup()  # 每个块处理后积极清理显存

            except Exception as e:
                print(f"⚠️ Error processing chunk {chunk_idx + 1}: {e}")
                continue

        # 合并所有块
        print("🔗 Merging chunks...")

        if use_streaming_merge:
            return self._merge_from_files(temp_files, sample_rate, Path(audio_path).stem)
        else:
            return self._merge_from_memory(accumulated_sources, sample_rate, Path(audio_path).stem)

    def _merge_from_memory(self, accumulated_sources, sample_rate: int, audio_name: str):
        """从内存中合并音频块"""
        source_names = self.model.sources
        final_sources = {}

        for source_name in source_names:
            if accumulated_sources[source_name]:
                try:
                    merged = torch.cat(accumulated_sources[source_name], dim=1)
                    final_sources[source_name] = merged
                    print(f"✅ Successfully merged {source_name}: {len(accumulated_sources[source_name])} chunks")

                    # 清理内存
                    del accumulated_sources[source_name]
                    gc.collect()

                except Exception as e:
                    print(f"⚠️ Failed to merge {source_name}: {e}")
                    continue
            else:
                print(f"⚠️ No chunks found for {source_name}")

        if not final_sources:
            raise RuntimeError("No audio sources were successfully processed")

        return self._save_sources_dict(final_sources, sample_rate, audio_name)

    def _merge_from_files(self, temp_files, sample_rate: int, audio_name: str):
        """从临时文件合并音频块 - 避免大内存占用"""
        source_names = self.model.sources
        outputs = {}

        for source_name in source_names:
            if temp_files[source_name]:
                try:
                    print(f"🔗 Merging {source_name} from {len(temp_files[source_name])} files...")

                    # 创建最终输出文件
                    final_file = Path(tempfile.gettempdir()) / f"demucs_{source_name}_{audio_name}.wav"

                    # 使用FFmpeg进行文件级合并，避免大内存占用
                    if len(temp_files[source_name]) == 1:
                        # 只有一个文件，直接重命名
                        shutil.move(temp_files[source_name][0], str(final_file))
                    else:
                        # 多个文件，使用FFmpeg合并
                        self._concat_audio_files(temp_files[source_name], str(final_file))

                    outputs[source_name] = str(final_file)
                    print(f"✅ Successfully merged {source_name}")

                    # 清理临时文件
                    for temp_file in temp_files[source_name]:
                        try:
                            if os.path.exists(temp_file) and temp_file != str(final_file):
                                os.remove(temp_file)
                        except Exception as e:
                            print(f"⚠️ Could not remove temp file {temp_file}: {e}")

                except Exception as e:
                    print(f"⚠️ Failed to merge {source_name}: {e}")
                    # 清理失败的临时文件
                    for temp_file in temp_files[source_name]:
                        try:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                        except:
                            pass
                    continue
            else:
                print(f"⚠️ No files found for {source_name}")

        if not outputs:
            raise RuntimeError("No audio sources were successfully processed")

        return outputs

    def _concat_audio_files(self, file_list, output_file):
        """使用FFmpeg连接音频文件"""
        try:
            # 创建临时文件列表
            list_file = Path(tempfile.gettempdir()) / f"concat_list_{os.getpid()}.txt"

            with open(list_file, 'w', encoding='utf-8') as f:
                for file_path in file_list:
                    # 使用绝对路径并转义特殊字符
                    abs_path = os.path.abspath(file_path).replace('\\', '/')
                    f.write(f"file '{abs_path}'\n")

            # 使用FFmpeg连接文件
            (
                ffmpeg
                .input(str(list_file), format='concat', safe=0)
                .output(output_file, acodec='pcm_s16le')
                .overwrite_output()
                .run(quiet=True, capture_stdout=True)
            )

            # 清理临时列表文件
            if os.path.exists(list_file):
                os.remove(list_file)

        except Exception as e:
            # 如果FFmpeg失败，回退到PyTorch方法
            print(f"⚠️ FFmpeg concat failed, using PyTorch fallback: {e}")
            self._concat_audio_files_pytorch(file_list, output_file)

    def _concat_audio_files_pytorch(self, file_list, output_file):
        """使用PyTorch连接音频文件（回退方法）"""
        try:
            # 逐个加载并写入，避免大内存占用
            first_file = True

            for i, file_path in enumerate(file_list):
                waveform, sample_rate = torchaudio.load(file_path)

                if first_file:
                    # 第一个文件，创建新文件
                    torchaudio.save(output_file, waveform, sample_rate)
                    first_file = False
                else:
                    # 后续文件，追加到现有文件
                    # 注意：torchaudio不支持直接追加，需要先读取现有文件
                    existing_waveform, _ = torchaudio.load(output_file)
                    combined = torch.cat([existing_waveform, waveform], dim=1)
                    torchaudio.save(output_file, combined, sample_rate)

                    # 清理内存
                    del existing_waveform, combined

                del waveform
                gc.collect()

        except Exception as e:
            raise RuntimeError(f"Failed to concatenate audio files: {e}")

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
        'ultra_high': {'bitrate': '320k', 'samplerate': '48000', 'description': '超高质量 - 16GB显卡专用'},
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
        """智能质量选择 - 16GB显卡优化版本"""
        if not self.auto_quality:
            return requested_quality

        info = self._get_video_info(video_path)
        duration_hours = info['duration'] / 3600
        file_size_mb = info['file_size_mb']

        print(f"📊 Video analysis: {duration_hours:.1f}h, {file_size_mb:.1f}MB")

        # 检查是否为16GB显卡
        is_16gb_gpu = False
        try:
            import torch
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                is_16gb_gpu = total_memory >= 15
        except:
            pass

        # 16GB显卡的智能质量选择规则
        if is_16gb_gpu:
            if duration_hours > 4.0:  # 超过4小时的超长视频
                recommended_quality = 'high'
                reason = f"16GB显卡长视频 ({duration_hours:.1f}h) - 高质量处理"
                time_saved = "大块处理，质量优先"
            elif duration_hours > 2.0:  # 超过2小时的长视频
                recommended_quality = 'ultra_high'
                reason = f"16GB显卡中长视频 ({duration_hours:.1f}h) - 超高质量"
                time_saved = "充分利用16GB显存"
            elif duration_hours > 0.5:  # 超过30分钟的视频
                recommended_quality = 'ultra_high'
                reason = f"16GB显卡标准视频 ({duration_hours:.1f}h) - 超高质量"
                time_saved = "最佳音质体验"
            else:  # 短视频
                recommended_quality = 'ultra_high'
                reason = f"16GB显卡短视频 ({duration_hours:.1f}h) - 超高质量"
                time_saved = "快速高质量处理"
        else:
            # 其他显卡的平衡质量与速度规则
            if duration_hours > 3.0:  # 超过3小时的长视频
                recommended_quality = 'medium'
                reason = f"长视频 ({duration_hours:.1f}h) - 平衡质量与处理时间"
                time_saved = "节省约60%处理时间"
            elif duration_hours > 1.5:  # 超过1.5小时的中长视频
                recommended_quality = 'medium'
                reason = f"中长视频 ({duration_hours:.1f}h) - 适中质量，合理时间"
                time_saved = "节省约50%处理时间"
            elif file_size_mb > 1000:  # 超过1GB的大文件
                recommended_quality = 'medium'
                reason = f"大文件 ({file_size_mb:.0f}MB) - 优化处理效率"
                time_saved = "节省约40%处理时间"
            else:
                recommended_quality = requested_quality  # 短视频保持高质量
                reason = f"短视频 ({duration_hours:.1f}h, {file_size_mb:.0f}MB) - 保持高质量"
                time_saved = ""

        # 给出调整建议
        if recommended_quality != requested_quality:
            print(f"🎯 智能质量优化: {requested_quality} → {recommended_quality}")
            print(f"   原因: {reason}")
            print(f"   说明: {self.QUALITY_SETTINGS[recommended_quality]['description']}")
            print(f"   ⚡ 效果: {time_saved}，质量仍然很好")
            return recommended_quality
        else:
            print(f"✅ 保持高质量设置: {requested_quality} ({reason})")
            return requested_quality

    def _estimate_processing_time(self, duration_hours: float, quality: str) -> str:
        """估算处理时间 - 平衡质量与速度的现实估算"""
        # 基于实际测试的处理时间估算（分钟）
        base_time = {
            'ultra_low': duration_hours * 6,   # 6分钟/小时
            'low': duration_hours * 10,        # 10分钟/小时
            'medium': duration_hours * 15,     # 15分钟/小时 (推荐的平衡选择)
            'high': duration_hours * 25        # 25分钟/小时 (高质量但耗时)
        }

        estimated_minutes = base_time.get(quality, duration_hours * 15)

        # 考虑各种优化的加速效果
        speedup_factor = 1.0

        # 半精度优化
        if hasattr(self, 'use_half_precision') and self.use_half_precision:
            speedup_factor *= 0.8  # 半精度提速20%

        # GPU显存充足时的额外加速
        if hasattr(self, 'max_memory_gb') and self.max_memory_gb >= 10:
            speedup_factor *= 0.9  # 大显存提速10%

        estimated_minutes *= speedup_factor

        # 生成友好的时间显示
        if estimated_minutes < 1:
            return "< 1分钟"
        elif estimated_minutes < 60:
            time_str = f"约 {estimated_minutes:.0f}分钟"
        else:
            hours = estimated_minutes / 60
            if hours < 1.5:
                time_str = f"约 {estimated_minutes:.0f}分钟"
            else:
                time_str = f"约 {hours:.1f}小时"

        # 添加质量说明
        quality_note = {
            'medium': " (推荐：质量好，速度快)",
            'high': " (高质量，较慢)",
            'low': " (快速处理)",
            'ultra_low': " (最快速度)"
        }.get(quality, "")

        return time_str + quality_note

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