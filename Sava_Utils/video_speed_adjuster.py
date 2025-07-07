"""
视频变速处理工具
根据原字幕和新字幕的时长差异对视频进行变速处理
"""

import gc
import os
import re
import datetime
import hashlib
import shutil
from typing import Dict, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import ffmpeg
import platform
import subprocess


class VideoSpeedAdjuster:
    """视频变速处理器"""

    def __init__(self, output_dir: str = None, max_workers: int = 4, use_gpu: bool = True):
        """
        初始化视频变速处理器

        Args:
            output_dir: 输出目录
            max_workers: 最大并发处理数
            use_gpu: 是否使用GPU加速
        """
        if output_dir is not None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(exist_ok=True)

        self.max_workers = max_workers
        self.session_id = hashlib.md5(f"{datetime.datetime.now().isoformat()}".encode()).hexdigest()[:8]
        self.temp_dir = Path(f"temp_video_{self.session_id}")
        self.temp_dir.mkdir(exist_ok=True)

        # 检测GPU编码器
        self.gpu_encoder = self._detect_gpu_encoder() if use_gpu else None

        print(f"🚀 VideoSpeedAdjuster initialized")
        if hasattr(self, 'output_dir'):
            print(f"📁 Output: {self.output_dir}")
        else:
            print(f"📁 Output: Not specified")
        print(f"⚡ Workers: {self.max_workers}")
        if self.gpu_encoder:
            print(f"🎮 GPU: {self.gpu_encoder}")
        else:
            print(f"💻 CPU encoding")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        """清理临时文件"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                print(f"🧹 Cleaned up: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️ Cleanup failed: {e}")
        gc.collect()

    def _detect_gpu_encoder(self) -> Optional[str]:
        """检测GPU编码器"""
        encoders = ["h264_nvenc", "h264_qsv", "h264_amf"] if platform.system() == "Windows" else ["h264_nvenc"]

        for encoder in encoders:
            try:
                cmd = ["ffmpeg", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=1",
                       "-c:v", encoder, "-t", "1", "-f", "null", "-"]
                if subprocess.run(cmd, capture_output=True, timeout=3).returncode == 0:
                    return encoder
            except:
                continue
        return None

    def parse_srt_file(self, srt_path: str) -> List[Dict]:
        """
        解析SRT字幕文件
        
        Args:
            srt_path: SRT文件路径
            
        Returns:
            List[Dict]: 字幕条目列表
        """
        if not os.path.exists(srt_path):
            raise FileNotFoundError(f"SRT file not found: {srt_path}")

        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'cp1252']
        content = None

        for encoding in encodings:
            try:
                with open(srt_path, 'r', encoding=encoding) as file:
                    content = file.read()
                print(f"✅ SRT file loaded with {encoding} encoding: {Path(srt_path).name}")
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
                start_time = self._parse_timestamp(start_time)
                end_time = self._parse_timestamp(end_time)
                duration = (end_time - start_time).total_seconds()
                text = ' '.join(lines[2:])

                # 清理文本中的标记
                text = re.sub(r'[（\(][^）\)]*[）\)]', '', text).strip()
                text = text.replace('-', '').strip()

                subtitles.append({
                    'number': number,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'text': text,
                    'start_seconds': self._timestamp_to_seconds(start_time),
                    'end_seconds': self._timestamp_to_seconds(end_time)
                })
            except (ValueError, IndexError) as e:
                print(f"⚠️ Warning: Failed to parse subtitle block: {e}")
                continue

        print(f"✅ Parsed {len(subtitles)} subtitle entries from {Path(srt_path).name}")
        return subtitles

    def _parse_timestamp(self, timestamp_str: str) -> datetime.datetime:
        """解析SRT时间戳"""
        timestamp_str = timestamp_str.replace(',', '.')
        return datetime.datetime.strptime(timestamp_str, '%H:%M:%S.%f')

    def _timestamp_to_seconds(self, timestamp: datetime.datetime) -> float:
        """将时间戳转换为秒数"""
        return (timestamp.hour * 3600 + timestamp.minute * 60 +
                timestamp.second + timestamp.microsecond / 1000000)

    def get_video_duration(self, video_path: str) -> float:
        """
        获取视频总时长

        Args:
            video_path: 视频文件路径

        Returns:
            float: 视频时长（秒）
        """
        try:
            # 使用 FFmpeg 获取视频时长
            probe = ffmpeg.probe(video_path)
            duration = float(probe['streams'][0]['duration'])
            print(f"🎬 Video duration: {duration:.2f} seconds")
            return duration
        except Exception as e:
            raise RuntimeError(f"Failed to get video duration: {e}")

    def validate_subtitles(self, original_subs: List[Dict], new_subs: List[Dict]) -> bool:
        """
        验证两个字幕文件的兼容性
        
        Args:
            original_subs: 原字幕列表
            new_subs: 新字幕列表
            
        Returns:
            bool: 是否兼容
        """
        if len(original_subs) != len(new_subs):
            raise ValueError(f"Subtitle count mismatch: original={len(original_subs)}, new={len(new_subs)}")

        # 检查字幕编号是否对应
        for i, (orig, new) in enumerate(zip(original_subs, new_subs)):
            if orig['number'] != new['number']:
                print(f"⚠️ Warning: Subtitle number mismatch at index {i}: {orig['number']} vs {new['number']}")

        print(f"✅ Subtitle validation passed: {len(original_subs)} entries")
        return True

    def calculate_speed_segments(self, original_subs: List[Dict], new_subs: List[Dict],
                                 video_duration: float) -> List[Dict]:
        """
        计算每个片段的变速参数
        
        Args:
            original_subs: 原字幕列表
            new_subs: 新字幕列表
            video_duration: 视频总时长
            
        Returns:
            List[Dict]: 片段变速信息列表
        """
        segments = []
        segment_index = 0

        # 检查第一行字幕是否从0开始，如果不是则添加开头片段
        first_subtitle_start = original_subs[0]['start_seconds'] if original_subs else 0
        if first_subtitle_start > 0.1:  # 如果第一行字幕开始时间大于0.1秒
            segments.append({
                'index': segment_index,
                'subtitle_number': 0,  # 特殊标记为开头片段
                'start_time': 0,
                'end_time': first_subtitle_start,
                'original_duration': first_subtitle_start,
                'target_duration': first_subtitle_start,  # 开头片段保持原速度
                'speed_ratio': 1.0,
                'original_text': '[开头片段]',
                'new_text': '[开头片段]'
            })
            segment_index += 1
            print(f"📍 Added opening segment: 0s - {first_subtitle_start:.2f}s")

        for i in range(len(original_subs)):
            orig_sub = original_subs[i]
            new_sub = new_subs[i]

            # 计算原片段的时间范围（从当前字幕开始到下一行字幕开始）
            start_time = orig_sub['start_seconds']

            # 结束时间：下一个字幕的开始时间，或视频结束时间
            if i < len(original_subs) - 1:
                end_time = original_subs[i + 1]['start_seconds']
            else:
                end_time = video_duration

            original_duration = end_time - start_time

            # 目标时长：新字幕对应片段的时长（从当前新字幕开始到下一行新字幕开始，或到结束）
            new_start_time = new_sub['start_seconds']
            if i < len(new_subs) - 1:
                new_end_time = new_subs[i + 1]['start_seconds']
            else:
                # 最后一行：从开始时间到结束时间
                new_end_time = new_sub['end_seconds']

            target_duration = new_end_time - new_start_time

            # 计算变速比例：需要将原片段时长调整为新字幕片段的时长
            # speed_ratio = 原时长 / 目标时长
            if target_duration > 0:
                speed_ratio = original_duration / target_duration
                # 限制变速比例在合理范围内（0.25x - 4.0x）
                speed_ratio = max(0.25, min(4.0, speed_ratio))
            else:
                speed_ratio = 1.0
                print(f"⚠️ Warning: Zero target duration for subtitle {new_sub['number']}, using 1.0x speed")

            segments.append({
                'index': segment_index,
                'subtitle_number': orig_sub['number'],
                'start_time': start_time,
                'end_time': end_time,
                'original_duration': original_duration,
                'target_duration': target_duration,  # 改名为target_duration更清晰
                'speed_ratio': speed_ratio,
                'original_text': orig_sub['text'],
                'new_text': new_sub['text']
            })
            segment_index += 1

        # 验证视频完整性
        self._verify_video_completeness(segments, video_duration)

        print(f"✅ Calculated speed parameters for {len(segments)} segments")
        return segments

    def _verify_video_completeness(self, segments: List[Dict], video_duration: float):
        """验证视频切割的完整性"""
        print("🔍 Verifying video completeness...")

        if not segments:
            print("❌ No segments found!")
            return

        # 按开始时间排序
        sorted_segments = sorted(segments, key=lambda x: x['start_time'])

        # 检查是否从0开始
        first_start = sorted_segments[0]['start_time']
        if first_start > 0.1:
            print(f"❌ Gap at beginning: 0s - {first_start:.2f}s")
        else:
            print(f"✅ Starts from: {first_start:.2f}s")

        # 检查片段间是否有间隔
        total_coverage = 0
        gaps = []
        overlaps = []

        for i, segment in enumerate(sorted_segments):
            start = segment['start_time']
            end = segment['end_time']
            duration = end - start
            total_coverage += duration

            print(f"   Segment {i + 1}: {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s)")

            # 检查与下一个片段的连接
            if i < len(sorted_segments) - 1:
                next_start = sorted_segments[i + 1]['start_time']
                if end < next_start - 0.01:  # 有间隔
                    gap_duration = next_start - end
                    gaps.append((end, next_start, gap_duration))
                    print(f"   ⚠️ Gap: {end:.2f}s - {next_start:.2f}s (duration: {gap_duration:.2f}s)")
                elif end > next_start + 0.01:  # 有重叠
                    overlap_duration = end - next_start
                    overlaps.append((next_start, end, overlap_duration))
                    print(f"   ⚠️ Overlap: {next_start:.2f}s - {end:.2f}s (duration: {overlap_duration:.2f}s)")

        # 检查是否覆盖到视频结束
        last_end = sorted_segments[-1]['end_time']
        if last_end < video_duration - 0.1:
            print(f"❌ Missing end: {last_end:.2f}s - {video_duration:.2f}s")
        else:
            print(f"✅ Ends at: {last_end:.2f}s (video: {video_duration:.2f}s)")

        # 总结
        print(f"📊 Coverage summary:")
        print(f"   Total segments: {len(segments)}")
        print(f"   Total coverage: {total_coverage:.2f}s")
        print(f"   Video duration: {video_duration:.2f}s")
        print(f"   Coverage ratio: {total_coverage / video_duration * 100:.1f}%")

        if gaps:
            print(f"   ❌ Found {len(gaps)} gaps")
        if overlaps:
            print(f"   ❌ Found {len(overlaps)} overlaps")

        if not gaps and not overlaps and abs(last_end - video_duration) < 0.1:
            print("✅ Video completeness verified!")
        else:
            print("❌ Video completeness issues detected!")

    def split_video_segment(self, video_path: str, start_time: float, end_time: float,
                            output_path: str) -> str:
        """分割无声视频片段"""
        try:
            duration = end_time - start_time
            input_stream = ffmpeg.input(video_path, ss=start_time, t=duration)

            # 选择编码器（无声视频，不需要音频编码）
            if self.gpu_encoder:
                output_stream = ffmpeg.output(input_stream, output_path, vcodec=self.gpu_encoder, **{'b:v': '2M'})
            else:
                output_stream = ffmpeg.output(input_stream, output_path, vcodec='libx264', crf=23)

            ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
            return output_path

        except Exception as e:
            raise RuntimeError(f"Split failed: {e}")

    def adjust_video_speed(self, input_path: str, output_path: str, speed_ratio: float) -> str:
        """调整无声视频播放速度"""
        try:
            # 接近1.0倍速直接复制
            if abs(speed_ratio - 1.0) < 0.01:
                shutil.copy2(input_path, output_path)
                return output_path

            # 调整视频速度（无声视频，只需处理视频流）
            input_stream = ffmpeg.input(input_path)
            video_stream = input_stream.video.filter('setpts', f'{1 / speed_ratio:.6f}*PTS')

            if self.gpu_encoder:
                output_stream = ffmpeg.output(video_stream, output_path, vcodec=self.gpu_encoder, **{'b:v': '2M'})
            else:
                output_stream = ffmpeg.output(video_stream, output_path, vcodec='libx264', crf=23)

            ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
            return output_path

        except Exception as e:
            raise RuntimeError(f"Speed adjust failed: {e}")

    def merge_video_audio(self, video_path: str, audio_path: str, output_path: str,
                          sync_to_audio: bool = True) -> str:
        """
        合成无声视频和音频为完整视频（优化版）

        Args:
            video_path: 无声视频文件路径（同时作为参考视频获取格式信息）
            audio_path: 音频文件路径
            output_path: 输出视频路径
            sync_to_audio: 是否将视频时长同步到音频时长

        Returns:
            str: 输出文件路径
        """
        try:
            print(f"🎬 Merging video and audio (optimized)...")
            print(f"   Video: {Path(video_path).name}")
            print(f"   Audio: {Path(audio_path).name}")

            # 验证输入文件
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # 获取视频和音频时长
            video_duration = self.get_video_duration(video_path)
            audio_duration = self._get_audio_duration(audio_path)

            print(f"⏱️ Video duration: {video_duration:.2f}s")
            print(f"⏱️ Audio duration: {audio_duration:.2f}s")

            # 获取参考视频信息和编码参数（使用原视频作为参考）
            video_codec, audio_codec, encoding_params = self._get_encoding_params(video_path)

            # 构建FFmpeg流
            video_input = ffmpeg.input(video_path)
            audio_input = ffmpeg.input(audio_path)

            # 标准合成方式（让后处理来处理时长不匹配）
            print(f"🔄 Standard merge...")
            if self.gpu_encoder:
                output_stream = ffmpeg.output(
                    video_input, audio_input, output_path,
                    vcodec=self.gpu_encoder,
                    acodec='aac',
                    **{'b:v': '2M'}
                )
            else:
                output_stream = ffmpeg.output(
                    video_input, audio_input, output_path,
                    vcodec='libx264',
                    acodec='aac',
                    crf=23
                )

            # 执行合成
            ffmpeg.run(output_stream, overwrite_output=True, quiet=True)

            # 验证输出文件
            if os.path.exists(output_path):
                output_duration = self.get_video_duration(output_path)
                print(f"✅ Video-audio merge completed: {Path(output_path).name}")
                print(f"📊 Output duration: {output_duration:.2f}s")

                # 检查同步效果并进行后处理
                if sync_to_audio and audio_duration > output_duration + 0.5:
                    print(f"🔄 Video shorter than audio, extending video...")
                    success = self._extend_video_to_audio_length(output_path, audio_duration)
                    if success:
                        output_duration = self.get_video_duration(output_path)
                        print(f"✅ Video extended to: {output_duration:.2f}s")

                sync_diff = abs(output_duration - audio_duration)
                if sync_diff < 0.1:
                    print(f"🎯 Perfect sync: ±{sync_diff:.2f}s")
                else:
                    print(f"⚠️ Sync difference: {sync_diff:.2f}s")
            else:
                raise RuntimeError("Output file was not created")

            return output_path

        except Exception as e:
            raise RuntimeError(f"Failed to merge video and audio: {e}")

    def _extend_video_to_audio_length(self, video_path: str, target_duration: float) -> bool:
        """
        延长视频到指定时长（通过重复最后一帧）

        Args:
            video_path: 输入视频路径
            target_duration: 目标时长

        Returns:
            bool: 是否成功
        """
        try:
            extended_path = video_path.replace('.mp4', '_extended.mp4')

            # 获取当前视频时长
            current_duration = self.get_video_duration(video_path)
            extend_duration = target_duration - current_duration

            print(
                f"   Current: {current_duration:.2f}s, Target: {target_duration:.2f}s, Extend: {extend_duration:.2f}s")

            # 使用简单的方法：创建一个静态图像并与原视频拼接
            temp_image = self.temp_dir / "last_frame.png"

            # 提取最后一帧
            (
                ffmpeg
                .input(video_path, ss=current_duration - 0.1)
                .output(str(temp_image), vframes=1)
                .run(overwrite_output=True, quiet=True)
            )

            # 创建延长的静态视频片段
            temp_static = self.temp_dir / "static_extension.mp4"
            print(f"   Creating {extend_duration:.2f}s extension...")
            (
                ffmpeg
                .input(str(temp_image), loop=1, t=extend_duration, framerate=25)
                .output(str(temp_static), vcodec='libx264', pix_fmt='yuv420p')
                .run(overwrite_output=True, quiet=True)
            )

            # 拼接原视频和延长片段
            filelist_path = self.temp_dir / "extend_filelist.txt"
            with open(filelist_path, 'w', encoding='utf-8') as f:
                f.write(f"file '{os.path.abspath(video_path).replace(chr(92), '/')}'\n")
                f.write(f"file '{os.path.abspath(temp_static).replace(chr(92), '/')}'\n")

            (
                ffmpeg
                .input(str(filelist_path), format='concat', safe=0)
                .output(extended_path, vcodec='copy')
                .run(overwrite_output=True, quiet=True)
            )

            # 清理临时文件
            for temp_file in [temp_image, temp_static, filelist_path]:
                if temp_file.exists():
                    temp_file.unlink()

            # 替换原文件
            if os.path.exists(extended_path):
                os.replace(extended_path, video_path)
                return True
            else:
                return False

        except Exception as e:
            print(f"⚠️ Failed to extend video: {e}")
            return False

    def _get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长"""
        try:
            probe = ffmpeg.probe(audio_path)
            for stream in probe['streams']:
                if stream['codec_type'] == 'audio':
                    return float(stream['duration'])
            raise RuntimeError("No audio stream found")
        except Exception as e:
            raise RuntimeError(f"Failed to get audio duration: {e}")

    def _get_encoding_params(self, reference_video_path: str = None) -> tuple:
        """获取编码参数"""
        video_codec = 'libx264'
        audio_codec = 'aac'
        encoding_params = {}

        # 从参考视频获取编码信息
        if reference_video_path and os.path.exists(reference_video_path):
            try:
                probe = ffmpeg.probe(reference_video_path)
                for stream in probe['streams']:
                    if stream['codec_type'] == 'video':
                        ref_codec = stream['codec_name']
                        if ref_codec in ['h264', 'libx264']:
                            video_codec = 'libx264'
                        elif ref_codec in ['h265', 'hevc', 'libx265']:
                            video_codec = 'libx265'

                        # 获取更多参数
                        if 'bit_rate' in stream:
                            bitrate = int(stream['bit_rate']) // 1000  # 转换为kbps
                            encoding_params['b:v'] = f'{min(bitrate, 5000)}k'  # 限制最大比特率
                        break

                print(f"📋 Reference codec: {ref_codec} → Using: {video_codec}")
            except Exception as e:
                print(f"⚠️ Could not read reference video info: {e}")

        # GPU编码器特定参数
        if self.gpu_encoder:
            if self.gpu_encoder == "h264_nvenc":
                encoding_params.update({
                    'preset': 'fast',
                    'profile:v': 'high',
                    'level': '4.1'
                })
            elif self.gpu_encoder == "h264_qsv":
                encoding_params.update({
                    'preset': 'medium',
                    'profile:v': 'high'
                })

        # 如果没有设置比特率，使用默认值
        if 'b:v' not in encoding_params:
            encoding_params['b:v'] = '2M'

        return video_codec, audio_codec, encoding_params

    def process_segment(self, video_path: str, segment: Dict, segment_dir: Path) -> Dict:
        """
        处理单个视频片段

        Args:
            video_path: 原视频路径
            segment: 片段信息
            segment_dir: 片段输出目录

        Returns:
            Dict: 处理结果
        """
        try:
            index = segment['index']
            start_time = segment['start_time']
            end_time = segment['end_time']
            speed_ratio = segment['speed_ratio']

            # 生成临时文件名
            temp_segment = segment_dir / f"temp_segment_{index:04d}.mp4"
            final_segment = segment_dir / f"segment_{index:04d}.mp4"

            original_dur = segment['original_duration']
            target_dur = segment['target_duration']
            print(
                f"🎬 Segment {index + 1}: {start_time:.2f}s-{end_time:.2f}s (原{original_dur:.2f}s→目标{target_dur:.2f}s, {speed_ratio:.2f}x)")

            # 步骤1: 分割视频片段
            self.split_video_segment(video_path, start_time, end_time, str(temp_segment))

            # 步骤2: 调整播放速度
            if abs(speed_ratio - 1.0) > 0.01:  # 只有当速度比例不是1.0时才调整
                self.adjust_video_speed(str(temp_segment), str(final_segment), speed_ratio)
                # 删除临时文件
                temp_segment.unlink()
            else:
                # 如果不需要变速，直接重命名
                temp_segment.rename(final_segment)

            return {
                'index': index,
                'success': True,
                'output_path': str(final_segment),
                'speed_ratio': speed_ratio,
                'message': f"Segment {index + 1} processed successfully"
            }

        except Exception as e:
            return {
                'index': segment['index'],
                'success': False,
                'output_path': None,
                'speed_ratio': segment['speed_ratio'],
                'message': f"Failed to process segment {segment['index'] + 1}: {e}"
            }

    def concatenate_videos(self, segment_paths: List[str], output_path: str) -> str:
        """拼接无声视频片段"""
        try:
            print(f"🔗 Concatenating {len(segment_paths)} segments...")

            # 创建文件列表
            filelist_path = self.temp_dir / "filelist.txt"
            with open(filelist_path, 'w', encoding='utf-8') as f:
                for path in segment_paths:
                    abs_path = os.path.abspath(path).replace('\\', '/')
                    f.write(f"file '{abs_path}'\n")

            # 尝试直接拼接
            try:
                input_stream = ffmpeg.input(str(filelist_path), format='concat', safe=0)
                output_stream = ffmpeg.output(input_stream, output_path, vcodec='copy')
                ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
                print(f"✅ Concatenated (stream copy)")
            except Exception:
                # 重编码拼接
                print("⚠️ Re-encoding...")
                input_stream = ffmpeg.input(str(filelist_path), format='concat', safe=0)

                if self.gpu_encoder:
                    output_stream = ffmpeg.output(input_stream, output_path, vcodec=self.gpu_encoder, **{'b:v': '2M'})
                else:
                    output_stream = ffmpeg.output(input_stream, output_path, vcodec='libx264', crf=23)

                ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
                print(f"✅ Concatenated ({'GPU' if self.gpu_encoder else 'CPU'})")

            return output_path

        except Exception as e:
            raise RuntimeError(f"Concatenate failed: {e}")

    def process_video_with_speed_adjustment(self, video_path: str, original_srt_path: str,
                                            new_srt_path: str, output_path: str = None) -> Dict:
        """主处理函数：根据字幕差异对无声视频进行变速处理"""
        try:
            print(f"🚀 Processing: {Path(video_path).name}")

            # 验证文件存在
            for file_path in [video_path, original_srt_path, new_srt_path]:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")

            # 解析字幕
            original_subs = self.parse_srt_file(original_srt_path)
            new_subs = self.parse_srt_file(new_srt_path)
            self.validate_subtitles(original_subs, new_subs)

            # 计算变速参数
            video_duration = self.get_video_duration(video_path)
            segments = self.calculate_speed_segments(original_subs, new_subs, video_duration)

            # 并行处理片段
            print(f"⚡ Processing {len(segments)} segments...")
            segment_dir = self.temp_dir / "segments"
            segment_dir.mkdir(exist_ok=True)

            processed_segments = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_segment = {
                    executor.submit(self.process_segment, video_path, segment, segment_dir): segment
                    for segment in segments
                }

                for future in as_completed(future_to_segment):
                    result = future.result()
                    if result['success']:
                        processed_segments.append(result)
                    else:
                        print(f"❌ {result['message']}")

            if len(processed_segments) != len(segments):
                raise RuntimeError(f"Failed segments: {len(segments) - len(processed_segments)}")

            # 拼接视频
            processed_segments.sort(key=lambda x: x['index'])
            segment_paths = [seg['output_path'] for seg in processed_segments]

            if output_path is None:
                output_path = self.output_dir / f"{Path(video_path).stem}_speed_adjusted.mp4"

            final_output = self.concatenate_videos(segment_paths, str(output_path))

            # 统计信息
            total_original = sum(seg['original_duration'] for seg in segments)
            total_target = sum(seg['target_duration'] for seg in segments)
            avg_speed = total_original / total_target if total_target > 0 else 1.0

            print(f"🎉 Completed! {len(processed_segments)}/{len(segments)} segments")
            print(f"⏱️ Original: {total_original:.2f}s → Target: {total_target:.2f}s (avg {avg_speed:.2f}x)")
            print(f"💾 Output: {final_output}")

            return {
                'success': True,
                'output_path': final_output,
                'segments_processed': len(processed_segments),
                'total_segments': len(segments),
                'original_duration': total_original,
                'target_duration': total_target,  # 改为target_duration
                'average_speed_ratio': avg_speed,
                'message': "Success"
            }

        except Exception as e:
            print(f"❌ Failed: {e}")
            return {
                'success': False,
                'output_path': None,
                'segments_processed': 0,
                'total_segments': 0,
                'message': str(e)
            }


def adjust_video_speed_by_subtitles(video_path: str, original_srt_path: str, new_srt_path: str,
                                    output_path: str = None, output_dir: str = "output",
                                    max_workers: int = 4, use_gpu: bool = True) -> Dict:
    """
    根据字幕差异调整无声视频播放速度

    Args:
        video_path: 无声视频文件路径
        original_srt_path: 原字幕文件路径
        new_srt_path: 新字幕文件路径
        output_path: 输出视频路径（可选）
        output_dir: 输出目录
        max_workers: 最大并发处理数
        use_gpu: 是否使用GPU加速

    Returns:
        Dict: 处理结果
    """
    with VideoSpeedAdjuster(output_dir=output_dir, max_workers=max_workers, use_gpu=use_gpu) as adjuster:
        return adjuster.process_video_with_speed_adjustment(
            video_path, original_srt_path, new_srt_path, output_path
        )


def merge_video_with_audio(video_path: str, audio_path: str, output_path: str,
                           use_gpu: bool = True, sync_to_audio: bool = True) -> str:
    """
    合成无声视频和音频的便捷函数（优化版）

    Args:
        video_path: 无声视频文件路径（同时作为参考视频获取格式信息）
        audio_path: 音频文件路径
        output_path: 输出视频路径
        use_gpu: 是否使用GPU加速
        sync_to_audio: 是否将视频时长同步到音频时长

    Returns:
        str: 输出文件路径
    """
    # 从输出路径获取输出目录
    output_dir = os.path.dirname(output_path)
    with VideoSpeedAdjuster(output_dir=output_dir, use_gpu=use_gpu) as adjuster:
        return adjuster.merge_video_audio(video_path, audio_path, output_path, sync_to_audio)
