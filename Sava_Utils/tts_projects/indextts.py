import os
import json

import gradio as gr
from gradio_client import Client, handle_file

from . import TTSProjet
from .. import logger, i18n
from ..subtitle_text_formatter import format_subtitle_text
import io
import soundfile as sf

# 可选导入音频处理库
try:
    import numpy as np
    import librosa

    AUDIO_DETECTION_AVAILABLE = True
except ImportError:
    AUDIO_DETECTION_AVAILABLE = False
    logger.warning("音频检测功能不可用：numpy 或 librosa 未安装。Clone 模式将使用基础选择逻辑。")

current_path = os.environ.get("current_path")


class IndexTTS(TTSProjet):
    def __init__(self, config):
        super().__init__("indextts", config)
        self.config_file = os.path.join(current_path, "SAVAdata", "indextts_config.json")
        self._ensure_config_dir()

    def _ensure_config_dir(self):
        """确保配置目录存在"""
        config_dir = os.path.dirname(self.config_file)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)

    def _load_config(self):
        """加载配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                return config
        except Exception as e:
            logger.warning(f"加载Index-TTS配置失败: {e}")

        # 返回默认配置
        return {
            "mode_selection": "内置",
            "builtin_audio_selection": "舒朗男声",
            "language": "中文",
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.8,
            "top_k": 30,
            "num_beams": 3,
            "repetition_penalty": 10.0,
            "length_penalty": 0.0,
            "max_mel_tokens": 600,
            "volume_gain": 1.0,
            "max_text_tokens_per_sentence": 120,
            "sentences_bucket_max_size": 4,
            "infer_mode": "普通推理",
            "api_url": "http://127.0.0.1:7860"
        }

    def _save_config(self, config):
        """保存配置"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存Index-TTS配置失败: {e}")

    def restore_config(self):
        """公共方法：恢复配置到UI组件"""
        try:
            current_config = self._load_config()
            logger.info(f"恢复Index-TTS配置到UI组件")

            # 更新所有组件的值
            if hasattr(self, 'mode_selection'):
                self.mode_selection.value = current_config.get("mode_selection", "内置")
            if hasattr(self, 'builtin_audio_selection'):
                self.builtin_audio_selection.value = current_config.get("builtin_audio_selection", "舒朗男声")
            if hasattr(self, 'language'):
                self.language.value = current_config.get("language", "中文")
            if hasattr(self, 'do_sample'):
                self.do_sample.value = current_config.get("do_sample", True)
            if hasattr(self, 'temperature'):
                self.temperature.value = current_config.get("temperature", 1.0)
            if hasattr(self, 'top_p'):
                self.top_p.value = current_config.get("top_p", 0.8)
            if hasattr(self, 'volume_gain'):
                self.volume_gain.value = current_config.get("volume_gain", 1.0)
            if hasattr(self, 'api_url'):
                self.api_url.value = current_config.get("api_url", "http://127.0.0.1:7860")

            # 更新组件可见性
            mode = current_config.get("mode_selection", "内置")
            if hasattr(self, 'builtin_audio_selection'):
                self.builtin_audio_selection.visible = (mode == "内置")
            if hasattr(self, 'builtin_audio_preview'):
                self.builtin_audio_preview.visible = (mode == "内置")
            if hasattr(self, 'reference_audio'):
                self.reference_audio.visible = (mode == "自定义")

            return True
        except Exception as e:
            logger.error(f"恢复配置失败: {e}")
            return False

    def get_builtin_audio_map(self):
        """获取内置音频映射表"""
        return {
            "舒朗男声": "builtin_audios/hunyin_6.mp3",
            "新闻女声": "builtin_audios/Chinese (Mandarin)_News_Anchor.mp3",
            "傲娇御姐": "builtin_audios/Chinese (Mandarin)_Mature_Woman.mp3",
            "不羁青年": "builtin_audios/Chinese (Mandarin)_Unrestrained_Young_Man.mp3",
            "嚣张小姐": "builtin_audios/Arrogant_Miss.mp3",
            "热心大婶": "builtin_audios/Chinese (Mandarin)_Kind-hearted_Antie.mp3",
            "港普空姐": "builtin_audios/Chinese (Mandarin)_HK_Flight_Attendant.mp3",
            "搞笑大爷": "builtin_audios/Chinese (Mandarin)_Humorous_Elder.mp3",
            "温润男声": "builtin_audios/Chinese (Mandarin)_Gentleman.mp3",
            "温暖闺蜜": "builtin_audios/Chinese (Mandarin)_Warm_Bestie.mp3",
            "播报男声": "builtin_audios/Chinese (Mandarin)_Male_Announcer.mp3",
            "甜美女声": "builtin_audios/Chinese (Mandarin)_Sweet_Lady.mp3",
            "南方小哥": "builtin_audios/Chinese (Mandarin)_Southern_Young_Man.mp3",
            "阅历姐姐": "builtin_audios/Chinese (Mandarin)_Wise_Women.mp3",
            "温润青年": "builtin_audios/Chinese (Mandarin)_Gentle_Youth.mp3",
            "温暖少女": "builtin_audios/Chinese (Mandarin)_Warm_Girl.mp3",
            "花甲奶奶": "builtin_audios/Chinese (Mandarin)_Kind-hearted_Elder.mp3",
            "憨憨萌兽": "builtin_audios/Chinese (Mandarin)_Cute_Spirit.mp3",
            "电台男主播": "builtin_audios/Chinese (Mandarin)_Radio_Host.mp3",
            "抒情男声": "builtin_audios/Chinese (Mandarin)_Lyrical_Voice.mp3",
            "率真弟弟": "builtin_audios/Chinese (Mandarin)_Straightforward_Boy.mp3",
            "真诚青年": "builtin_audios/Chinese (Mandarin)_Sincere_Adult.mp3",
            "温柔学姐": "builtin_audios/Chinese (Mandarin)_Gentle_Senior.mp3",
            "嘴硬竹马": "builtin_audios/Chinese (Mandarin)_Stubborn_Friend.mp3",
            "清脆少女": "builtin_audios/Chinese (Mandarin)_Crisp_Girl.mp3",
            "清澈邻家弟弟": "builtin_audios/Chinese (Mandarin)_Pure-hearted_Boy.mp3",
            "软软女孩": "builtin_audios/Chinese (Mandarin)_Soft_Girl.mp3"
        }

    def get_default_builtin_audio(self):
        """获取默认的内置音频文件路径"""
        builtin_audio_map = self.get_builtin_audio_map()
        # 使用默认选择的音频（舒朗男声）
        default_audio_name = "舒朗男声"
        audio_path = builtin_audio_map.get(default_audio_name)

        if audio_path and os.path.exists(audio_path):
            return audio_path
        else:
            # 如果默认音频不存在，尝试使用第一个可用的音频
            for name, path in builtin_audio_map.items():
                if os.path.exists(path):
                    return path

            # 如果都不存在，返回None
            return None

    def _get_clone_reference_audio(self, subtitle_index=None):
        """
        获取clone模式的参考音频
        检查视频是否已加载，如果未加载则提示用户先加载视频
        如果已加载，根据字幕行号使用对应的音频片段作为参考音频

        Args:
            subtitle_index: 字幕行号，用于确定使用哪个segment文件
        """
        try:
            # 检查是否有当前视频路径环境变量
            current_video_path = os.environ.get("current_video_path")
            if not current_video_path or not os.path.exists(current_video_path):
                return None

            # 检查是否有segments目录
            current_path = os.environ.get("current_path", ".")
            segments_dir = None

            # 查找segments目录（新结构: SAVAdata/temp/audio_processing/{workspace_name}/segments）
            audio_processing_base = os.path.join(current_path, "SAVAdata", "temp", "audio_processing")
            if os.path.exists(audio_processing_base):
                for workspace_dir in os.listdir(audio_processing_base):
                    workspace_dir_path = os.path.join(audio_processing_base, workspace_dir)
                    if os.path.isdir(workspace_dir_path):
                        segments_path = os.path.join(workspace_dir_path, "segments")
                        if os.path.exists(segments_path):
                            segments_dir = segments_path
                            break

            if not segments_dir:
                logger.warning("未找到segments目录，请先加载视频文件进行音频分割")
                return None

            # 智能选择参考音频
            return self._smart_select_reference_audio(segments_dir, subtitle_index)

        except Exception as e:
            logger.error(f"获取clone参考音频时出错: {str(e)}")
            return None

    def _smart_select_reference_audio(self, segments_dir, subtitle_index):
        """
        智能选择参考音频，优先使用对应片段，如果无语音则选择最近的有效片段

        Args:
            segments_dir: segments目录路径
            subtitle_index: 字幕行号

        Returns:
            str: 选中的音频文件路径
        """
        try:
            # 如果音频检测不可用，使用基础选择逻辑
            if not AUDIO_DETECTION_AVAILABLE:
                return self._basic_select_reference_audio(segments_dir, subtitle_index)
            # 获取所有segment文件
            all_segments = []
            for file in os.listdir(segments_dir):
                if file.endswith(('.wav', '.mp3', '.flac', '.m4a')) and file.startswith('segment_'):
                    # 提取文件编号
                    try:
                        number_str = file.replace('segment_', '').split('.')[0]
                        number = int(number_str)
                        all_segments.append((number, file))
                    except ValueError:
                        continue

            if not all_segments:
                logger.error("segments目录中没有找到有效的音频文件")
                return None

            # 按编号排序
            all_segments.sort(key=lambda x: x[0])

            # 确定目标编号
            target_number = subtitle_index if subtitle_index is not None else 1

            # 首先尝试使用对应的片段
            target_file = None
            for number, filename in all_segments:
                if number == target_number:
                    target_file = os.path.join(segments_dir, filename)
                    break

            # 如果找到对应片段，检测是否有语音
            if target_file and os.path.exists(target_file):
                has_voice, energy, duration = self._detect_audio_activity(target_file)
                if has_voice:
                    logger.info(f"使用对应片段作为参考音频: {target_file} (行号: {target_number})")
                    return target_file
                else:
                    logger.warning(f"对应片段无有效语音: {target_file} (能量: {energy:.4f}, 时长: {duration:.2f}s)")

            # 如果对应片段无语音或不存在，寻找最近的有效片段
            logger.info(f"寻找最近的有效语音片段 (目标行号: {target_number})")

            # 使用双向搜索策略：优先考虑距离，同时考虑前后方向
            return self._find_nearest_valid_segment(all_segments, target_number, segments_dir)

        except Exception as e:
            logger.error(f"智能选择参考音频失败: {str(e)}")
            # 降级到基础选择
            return self._basic_select_reference_audio(segments_dir, subtitle_index)

    def _find_nearest_valid_segment(self, all_segments, target_number, segments_dir):
        """
        双向搜索最近的有效语音片段

        Args:
            all_segments: 所有片段列表 [(number, filename), ...]
            target_number: 目标行号
            segments_dir: segments目录路径

        Returns:
            str: 选中的音频文件路径
        """
        # 按距离分组：相同距离的片段放在一起
        distance_groups = {}
        for number, filename in all_segments:
            distance = abs(number - target_number)
            if distance not in distance_groups:
                distance_groups[distance] = []
            distance_groups[distance].append((number, filename))

        # 按距离从小到大检测
        for distance in sorted(distance_groups.keys()):
            segments_at_distance = distance_groups[distance]

            # 对于相同距离的片段，智能选择方向
            # 如果目标在末尾，优先选择前面的（编号更小的）
            if target_number > len(all_segments) * 0.8:  # 如果目标在后80%，优先选择前面的
                segments_at_distance.sort(key=lambda x: x[0])  # 从小到大（编号小的在前，即前面的片段）
                direction_hint = "向前搜索"
            else:
                segments_at_distance.sort(key=lambda x: x[0], reverse=True)  # 从大到小（编号大的在前，即后面的片段）
                direction_hint = "向后搜索"

            logger.debug(f"检测距离 {distance} 的片段 ({direction_hint}): {[x[0] for x in segments_at_distance]}")

            # 检测这个距离上的所有片段
            for number, filename in segments_at_distance:
                file_path = os.path.join(segments_dir, filename)

                if not AUDIO_DETECTION_AVAILABLE:
                    # 如果没有音频检测库，直接返回第一个找到的
                    logger.info(f"使用最近片段 (无音频检测): {file_path} (行号: {number}, 距离: {distance})")
                    return file_path

                has_voice, energy, duration = self._detect_audio_activity(file_path)

                if has_voice:
                    direction = "前面" if number < target_number else "后面" if number > target_number else "当前"
                    logger.info(f"找到有效语音片段: {file_path} (行号: {number}, 距离: {distance}, "
                                f"方向: {direction}, 能量: {energy:.4f}, 时长: {duration:.2f}s)")
                    return file_path
                else:
                    logger.debug(f"片段无有效语音: {file_path} (行号: {number}, 距离: {distance}, 能量: {energy:.4f})")

        # 如果所有片段都没有有效语音，使用启发式选择
        return self._fallback_segment_selection(all_segments, target_number, segments_dir)

    def _fallback_segment_selection(self, all_segments, target_number, segments_dir):
        """
        备选片段选择策略

        Args:
            all_segments: 所有片段列表
            target_number: 目标行号
            segments_dir: segments目录路径

        Returns:
            str: 选中的音频文件路径
        """
        logger.warning("所有片段都无有效语音，使用备选策略")

        # 策略1: 选择中间位置的片段（通常语音质量较好）
        middle_index = len(all_segments) // 2
        if middle_index < len(all_segments):
            middle_segment = all_segments[middle_index]
            middle_file = os.path.join(segments_dir, middle_segment[1])
            logger.info(f"使用中间片段作为备选: {middle_file} (行号: {middle_segment[0]})")
            return middle_file

        # 策略2: 如果没有中间片段，使用第一个
        if all_segments:
            first_segment = all_segments[0]
            first_file = os.path.join(segments_dir, first_segment[1])
            logger.info(f"使用第一个片段作为备选: {first_file} (行号: {first_segment[0]})")
            return first_file

        # 策略3: 如果连片段都没有，返回None
        logger.error("没有找到任何可用的音频片段")
        return None

    def _basic_select_reference_audio(self, segments_dir, subtitle_index):
        """
        基础参考音频选择（不依赖音频检测库）

        Args:
            segments_dir: segments目录路径
            subtitle_index: 字幕行号

        Returns:
            str: 选中的音频文件路径
        """
        try:
            # 获取所有segment文件
            all_segments = []
            for file in os.listdir(segments_dir):
                if file.endswith(('.wav', '.mp3', '.flac', '.m4a')) and file.startswith('segment_'):
                    try:
                        number_str = file.replace('segment_', '').split('.')[0]
                        number = int(number_str)
                        all_segments.append((number, file))
                    except ValueError:
                        continue

            if not all_segments:
                logger.error("segments目录中没有找到有效的音频文件")
                return None

            # 按编号排序
            all_segments.sort(key=lambda x: x[0])

            # 确定目标编号
            target_number = subtitle_index if subtitle_index is not None else 1

            # 首先尝试使用对应的片段
            for number, filename in all_segments:
                if number == target_number:
                    target_file = os.path.join(segments_dir, filename)
                    if os.path.exists(target_file):
                        logger.info(f"使用对应片段作为参考音频: {target_file} (行号: {target_number})")
                        return target_file

            # 如果对应片段不存在，使用双向搜索选择最近的片段
            return self._basic_find_nearest_segment(all_segments, target_number, segments_dir)

        except Exception as e:
            logger.error(f"基础选择参考音频失败: {str(e)}")
            return None

    def _basic_find_nearest_segment(self, all_segments, target_number, segments_dir):
        """
        基础双向搜索最近片段（不依赖音频检测）

        Args:
            all_segments: 所有片段列表
            target_number: 目标行号
            segments_dir: segments目录路径

        Returns:
            str: 选中的音频文件路径
        """
        # 按距离分组
        distance_groups = {}
        for number, filename in all_segments:
            distance = abs(number - target_number)
            if distance not in distance_groups:
                distance_groups[distance] = []
            distance_groups[distance].append((number, filename))

        # 按距离从小到大选择
        for distance in sorted(distance_groups.keys()):
            segments_at_distance = distance_groups[distance]

            # 智能方向选择
            if target_number > len(all_segments) * 0.8:  # 目标在后80%
                # 优先选择前面的片段
                selected = min(segments_at_distance, key=lambda x: x[0])
                direction = "向前"
            else:
                # 优先选择后面的片段
                selected = max(segments_at_distance, key=lambda x: x[0])
                direction = "向后"

            selected_file = os.path.join(segments_dir, selected[1])
            logger.info(f"使用最近片段: {selected_file} (行号: {selected[0]}, 距离: {distance}, "
                        f"目标: {target_number}, 方向: {direction})")
            return selected_file

        # 如果没有找到，返回None
        return None

    def _check_segments_exist(self):
        """
        检查 segments 目录是否存在且包含音频文件
        """
        try:
            current_path = os.environ.get("current_path", ".")

            # 新的目录结构是: SAVAdata/temp/audio_processing/{workspace_name}/segments
            audio_processing_base = os.path.join(current_path, "SAVAdata", "temp", "audio_processing")

            if not os.path.exists(audio_processing_base):
                return False

            # 查找 segments 目录
            for workspace_dir in os.listdir(audio_processing_base):
                workspace_dir_path = os.path.join(audio_processing_base, workspace_dir)
                if os.path.isdir(workspace_dir_path):
                    segments_path = os.path.join(workspace_dir_path, "segments")
                    if os.path.exists(segments_path):
                        # 检查是否有音频文件
                        for file in os.listdir(segments_path):
                            if file.endswith(('.wav', '.mp3', '.flac', '.m4a')):
                                return True
            return False
        except Exception as e:
            logger.error(f"检查segments时出错: {str(e)}")
            return False

    def _detect_audio_activity(self, audio_path):
        """
        检测音频文件是否包含有效的语音活动

        Args:
            audio_path: 音频文件路径

        Returns:
            tuple: (has_voice, energy_level, duration)
        """
        # 如果音频检测库不可用，返回默认值
        if not AUDIO_DETECTION_AVAILABLE:
            return True, 1.0, 1.0  # 假设有语音

        try:
            # 加载音频文件
            y, sr = librosa.load(audio_path, sr=None)

            # 计算音频时长
            duration = len(y) / sr

            # 如果音频太短（小于0.1秒），认为无效
            if duration < 0.1:
                return False, 0.0, duration

            # 计算RMS能量
            rms = librosa.feature.rms(y=y)[0]
            avg_energy = np.mean(rms)

            # 计算过零率（语音特征）
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            avg_zcr = np.mean(zcr)

            # 计算频谱质心（音色特征）
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            avg_centroid = np.mean(spectral_centroids)

            # 语音活动检测阈值（极宽松检测，保留任何可能的人声）
            energy_threshold = 0.005  # RMS能量阈值（极低，保留轻微人声）
            zcr_threshold = 0.01  # 过零率阈值（极低，保留各种语音特征）
            centroid_threshold = 300  # 频谱质心阈值（极低，保留所有频段语音）

            # 基础语音检测（宽松条件）
            basic_voice_check = (
                    avg_energy > energy_threshold and
                    avg_zcr > zcr_threshold and
                    avg_centroid > centroid_threshold
            )

            # 极简噪音过滤（只过滤最明显的电子噪音）
            # 几乎保留所有音频，只过滤明显的设备噪音
            is_short_noise = False

            if duration < 0.05:  # 极极短音频（小于0.05秒，仅过滤瞬间噪音）
                is_short_noise = True
            elif avg_zcr > 0.9:  # 过零率极极高（明显的电子噪音）
                is_short_noise = True

            # 最终判断：基础检测通过 AND 不是短暂噪音
            has_voice = basic_voice_check and not is_short_noise

            logger.debug(f"音频检测 {audio_path}: 能量={avg_energy:.4f}, 过零率={avg_zcr:.4f}, "
                         f"频谱质心={avg_centroid:.1f}, 时长={duration:.2f}s, "
                         f"基础检测={basic_voice_check}, 短暂噪音={is_short_noise}, 有语音={has_voice}")

            return has_voice, avg_energy, duration

        except Exception as e:
            logger.error(f"音频活动检测失败 {audio_path}: {str(e)}")
            return False, 0.0, 0.0

    def _apply_volume_gain(self, audio_data, volume_gain):
        """
        对音频数据应用音量增益

        Args:
            audio_data: 音频数据（bytes格式）
            volume_gain: 音量增益倍数

        Returns:
            bytes: 处理后的音频数据
        """
        try:
            # 如果音量增益为1.0，直接返回原音频
            if volume_gain == 1.0:
                return audio_data

            logger.info(f"正在应用音量增益: {volume_gain}x")

            # 将音频数据转换为numpy数组
            audio_io = io.BytesIO(audio_data)
            audio_array, sample_rate = sf.read(audio_io)

            # 应用音量增益
            audio_array = audio_array * volume_gain

            # 防止音频削波（限制在-1到1之间）
            audio_array = np.clip(audio_array, -1.0, 1.0)

            # 转换回音频数据
            output_io = io.BytesIO()
            sf.write(output_io, audio_array, sample_rate, format='WAV')
            output_io.seek(0)

            enhanced_audio = output_io.read()
            logger.info(f"音量增益应用成功，原始大小: {len(audio_data)} bytes，处理后大小: {len(enhanced_audio)} bytes")

            return enhanced_audio

        except Exception as e:
            logger.error(f"音量增益应用失败: {e}")
            # 如果处理失败，返回原始音频
            return audio_data

    def api(self, api_url, text, reference_audio, mode_selection, builtin_audio_selection, language, do_sample, top_k,
            top_p, temperature,
            num_beams, repetition_penalty, length_penalty, max_mel_tokens,
            max_text_tokens_per_sentence, sentences_bucket_max_size, infer_mode):
        """调用Index-TTS API"""
        try:
            # 确保 API URL 格式正确
            if not api_url.startswith('http://') and not api_url.startswith('https://'):
                api_url = f'http://{api_url}'

            # 创建Gradio客户端
            client = Client(api_url, httpx_kwargs={"timeout": 7200, "proxy": None}, ssl_verify=False)

            # 根据模式处理音频文件
            audio_file_path = None
            if mode_selection == "内置":
                # 使用内置音频，通过共享方法获取映射
                builtin_audio_map = self.get_builtin_audio_map()
                audio_file_path = builtin_audio_map.get(builtin_audio_selection)
                if not audio_file_path or not os.path.exists(audio_file_path):
                    logger.warning(f"内置音频文件不存在: {audio_file_path}, 使用默认音频")
                    # 这里可以设置一个默认的内置音频路径
                    audio_file_path = "builtin_audios/hunyin_6.mp3"
            elif mode_selection == "自定义":
                # 使用用户上传的音频文件
                if not reference_audio or not os.path.exists(reference_audio):
                    logger.error(f"参考音频文件不存在: {reference_audio}")
                    return None
                audio_file_path = reference_audio
            elif mode_selection == "clone":
                # clone模式根据字幕行号使用对应的segment作为参考音频
                logger.info("使用clone模式，获取参考音频...")

                # 获取当前字幕索引
                subtitle_index = None
                try:
                    subtitle_index_str = os.environ.get("current_subtitle_index")
                    if subtitle_index_str:
                        subtitle_index = int(subtitle_index_str)
                except (ValueError, TypeError):
                    logger.warning("无法获取字幕索引，将使用第一个segment")
                    subtitle_index = None

                audio_file_path = self._get_clone_reference_audio(subtitle_index)
                if audio_file_path is None:
                    logger.error("无法获取clone参考音频")
                    return None

            logger.info(f"Index-TTS API调用参数: text={text[:50]}..., mode={mode_selection}, "
                        f"builtin_audio={builtin_audio_selection if mode_selection == '内置' else 'N/A'}, "
                        f"language={language}, infer_mode={infer_mode}, do_sample={do_sample}, "
                        f"top_k={top_k}, top_p={top_p}, temperature={temperature}")

            # 调用Index-TTS的gen_single接口
            result = client.predict(
                handle_file(audio_file_path),  # prompt
                text,  # text
                infer_mode,  # infer_mode
                int(max_text_tokens_per_sentence),  # max_text_tokens_per_sentence
                int(sentences_bucket_max_size),  # sentences_bucket_max_size
                do_sample,  # do_sample
                float(top_p),  # top_p
                int(top_k) if int(top_k) > 0 else 0,  # top_k
                float(temperature),  # temperature
                float(length_penalty),  # length_penalty
                int(num_beams),  # num_beams
                float(repetition_penalty),  # repetition_penalty
                int(max_mel_tokens),  # max_mel_tokens
                api_name='/gen_single'
            )

            logger.info(f'Index-TTS result={result}')

            # 获取生成的音频文件路径 - result的value属性才是音频地址
            if hasattr(result, 'value'):
                wav_file = result.value
            elif isinstance(result, dict) and 'value' in result:
                wav_file = result['value']
            elif isinstance(result, (list, tuple)) and len(result) > 0:
                # 如果是列表/元组，取第一个元素的value属性
                first_item = result[0]
                if hasattr(first_item, 'value'):
                    wav_file = first_item.value
                elif isinstance(first_item, dict) and 'value' in first_item:
                    wav_file = first_item['value']
                else:
                    wav_file = first_item
            else:
                wav_file = result

            logger.info(f'解析得到的音频文件路径: {wav_file}')

            if wav_file and os.path.exists(wav_file):
                # 读取音频文件内容
                with open(wav_file, 'rb') as f:
                    audio_content = f.read()
                logger.info(f"成功生成音频文件: {wav_file}")
                return audio_content
            else:
                logger.error(f"生成的音频文件不存在: {wav_file}")
                return None

        except Exception as e:
            err = f"{i18n('An error has occurred. Please check if the API is running correctly. Details')}:{e}"
            logger.error(err)
            return None

    def save_action(self, *args, text: str = None):
        """保存操作，调用API并返回音频数据"""
        reference_audio, mode_selection, builtin_audio_selection, language, do_sample, top_k, top_p, temperature, num_beams, repetition_penalty, length_penalty, max_mel_tokens, volume_gain, max_text_tokens_per_sentence, sentences_bucket_max_size, infer_mode, api_url = args

        # 字幕文本格式化处理
        if text:
            original_text = text
            formatted_text = format_subtitle_text(text)
            if formatted_text != original_text:
                logger.info(f"字幕文本格式化: '{original_text}' → '{formatted_text}'")
                text = formatted_text

        audio = self.api(
            api_url=api_url,
            text=text,
            reference_audio=reference_audio,
            mode_selection=mode_selection,
            builtin_audio_selection=builtin_audio_selection,
            language=language,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            max_mel_tokens=max_mel_tokens,
            max_text_tokens_per_sentence=max_text_tokens_per_sentence,
            sentences_bucket_max_size=sentences_bucket_max_size,
            infer_mode=infer_mode
        )

        # 应用音量增益
        if audio and volume_gain != 1.0:
            audio = self._apply_volume_gain(audio, volume_gain)

        return audio

    def _UI(self):
        """创建Index-TTS的UI界面"""
        # 强制重新加载最新配置，不使用任何缓存
        self._ensure_config_dir()  # 确保配置目录存在
        saved_config = self._load_config()
        logger.info(f"UI初始化时强制加载最新配置: {saved_config}")

        with gr.TabItem("🔥 Index-TTS"):
            with gr.Column():
                # 参考音频模式 - 放在最上面，强制使用最新配置
                current_mode = self._load_config().get("mode_selection", "内置")
                logger.info(f"设置参考音频模式为: {current_mode}")

                self.mode_selection = gr.Radio(
                    label="参考音频模式",
                    choices=["内置", "clone", "自定义"],
                    value=current_mode,
                    interactive=True
                )

                # 内置音频选择 - 强制使用最新配置
                current_builtin_audio = self._load_config().get("builtin_audio_selection", "舒朗男声")
                current_builtin_visible = (current_mode == "内置")
                logger.info(f"设置内置音频选择: {current_builtin_audio}, 可见性: {current_builtin_visible}")

                self.builtin_audio_selection = gr.Dropdown(
                    label="内置音频选择",
                    choices=[
                        "舒朗男声", "新闻女声", "傲娇御姐",
                        "不羁青年", "嚣张小姐", "热心大婶",
                        "港普空姐", "搞笑大爷", "温润男声",
                        "温暖闺蜜", "播报男声", "甜美女声",
                        "南方小哥", "阅历姐姐", "温润青年",
                        "温暖少女", "花甲奶奶", "憨憨萌兽",
                        "电台男主播", "抒情男声", "率真弟弟",
                        "真诚青年", "温柔学姐", "嘴硬竹马",
                        "清脆少女", "清澈邻家弟弟", "软软女孩"
                    ],
                    value=current_builtin_audio,
                    visible=current_builtin_visible,
                    interactive=True
                )

                # 内置音频试听 - 强制使用最新配置
                current_preview_visible = (current_mode == "内置")
                logger.info(f"设置试听内置音频可见性: {current_preview_visible}")

                default_audio_path = self.get_default_builtin_audio()
                self.builtin_audio_preview = gr.Audio(
                    label="试听内置音频",
                    value=default_audio_path,
                    visible=current_preview_visible,
                    interactive=False
                )

                # 参考音频上传 - 强制使用最新配置
                current_reference_visible = (current_mode == "自定义")
                logger.info(f"设置参考音频上传可见性: {current_reference_visible}")

                self.reference_audio = gr.Audio(
                    label=i18n("Reference Audio"),
                    type="filepath",
                    visible=current_reference_visible
                )

                # 合成语言
                self.language = gr.Dropdown(
                    label=i18n("Inference text language"),
                    choices=["中文", "英文", "日文", "中英混合", "日英混合", "中英日混合"],
                    value=saved_config.get("language", "中文"),
                    interactive=True
                )

                with gr.Accordion("🔧 高级合成参数", open=False):
                    # 第一行：采样控制和音频控制
                    with gr.Row():
                        # 采样设置组
                        with gr.Group():
                            gr.Markdown("#### 🎯 采样控制")
                            self.do_sample = gr.Checkbox(
                                label="启用采样",
                                value=saved_config.get("do_sample", True),
                                interactive=True,
                                info="开启后使用随机采样，关闭则使用贪心搜索"
                            )
                            self.temperature = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                step=0.1,
                                value=saved_config.get("temperature", 1.0),
                                label="Temperature",
                                info="控制生成的随机性，值越高越随机"
                            )

                        # 音频生成控制组
                        with gr.Group():
                            gr.Markdown("#### 🎵 音频控制")
                            self.max_mel_tokens = gr.Slider(
                                minimum=50,
                                maximum=800,
                                step=10,
                                value=saved_config.get("max_mel_tokens", 600),
                                label="最大音频Token数",
                                info="控制生成音频的最大长度，过小会导致音频被截断"
                            )
                            self.volume_gain = gr.Slider(
                                minimum=0.1,
                                maximum=5.0,
                                step=0.1,
                                value=saved_config.get("volume_gain", 1.0),
                                label="音量增益",
                                info="音频音量倍数，1.0为原始音量，>1.0增强音量，最大5.0倍"
                            )

                    # 第二行：生成策略和惩罚机制
                    with gr.Row():
                        # 生成策略组
                        with gr.Group():
                            gr.Markdown("#### 🎲 生成策略")
                            self.top_p = gr.Slider(
                                minimum=0,
                                maximum=1,
                                step=0.01,
                                value=saved_config.get("top_p", 0.8),
                                label="Top-P",
                                info="核采样概率阈值，控制词汇选择范围"
                            )
                            self.top_k = gr.Slider(
                                minimum=0,
                                maximum=100,
                                step=1,
                                value=saved_config.get("top_k", 30),
                                label="Top-K",
                                info="保留概率最高的K个词汇"
                            )
                            self.num_beams = gr.Slider(
                                minimum=1,
                                maximum=10,
                                step=1,
                                value=saved_config.get("num_beams", 3),
                                label="Beam Size",
                                info="束搜索大小，值越大质量越高但速度越慢"
                            )

                        # 惩罚机制组
                        with gr.Group():
                            gr.Markdown("#### ⚖️ 惩罚机制")
                            self.repetition_penalty = gr.Slider(
                                minimum=1.0,
                                maximum=20.0,
                                step=0.1,
                                value=saved_config.get("repetition_penalty", 10.0),
                                label="重复惩罚",
                                info="防止重复生成，值越大惩罚越重"
                            )
                            self.length_penalty = gr.Slider(
                                minimum=-2.0,
                                maximum=2.0,
                                step=0.1,
                                value=saved_config.get("length_penalty", 0.0),
                                label="长度惩罚",
                                info="控制生成长度，正值偏好长句，负值偏好短句"
                            )

                    # 第三行：分句处理和推理模式
                    with gr.Row():
                        # 分句处理组
                        with gr.Group():
                            gr.Markdown("#### 📝 分句处理")
                            self.max_text_tokens_per_sentence = gr.Slider(
                                minimum=20,
                                maximum=600,
                                step=10,
                                value=saved_config.get("max_text_tokens_per_sentence", 120),
                                label="单句最大Token数",
                                info="推荐 20-300，值越大单次处理越长，过小会增加推理次数"
                            )
                            self.sentences_bucket_max_size = gr.Slider(
                                minimum=1,
                                maximum=16,
                                step=1,
                                value=saved_config.get("sentences_bucket_max_size", 4),
                                label="批次处理容量",
                                info="推荐 2-8，值越大批次越大，过大可能显存溢出"
                            )

                        # 推理模式组
                        with gr.Group():
                            gr.Markdown("#### 🚀 推理模式")
                            self.infer_mode = gr.Radio(
                                label="选择推理模式",
                                choices=["普通推理", "批次推理"],
                                value=saved_config.get("infer_mode", "普通推理"),
                                interactive=True,
                                info="批次推理速度更快但占用更多显存"
                            )

                # API服务地址
                self.api_url = gr.Textbox(
                    label="服务地址",
                    value=saved_config.get("api_url", "http://127.0.0.1:7860"),
                    placeholder="请输入Index-TTS服务地址，如: http://127.0.0.1:7860",
                    interactive=True
                )

                # 生成按钮
                self.gen_btn5 = gr.Button(value=i18n('Generate Audio'), variant="primary")

        # 添加动态显示逻辑
        def update_audio_components(mode):
            """根据参考音频模式更新组件显示"""
            if mode == "内置":
                return {
                    self.builtin_audio_selection: gr.update(visible=True),
                    self.builtin_audio_preview: gr.update(visible=True),
                    self.reference_audio: gr.update(visible=False)
                }
            elif mode == "自定义":
                return {
                    self.builtin_audio_selection: gr.update(visible=False),
                    self.builtin_audio_preview: gr.update(visible=False),
                    self.reference_audio: gr.update(visible=True)
                }
            else:  # clone
                return {
                    self.builtin_audio_selection: gr.update(visible=False),
                    self.builtin_audio_preview: gr.update(visible=False),
                    self.reference_audio: gr.update(visible=False)
                }

        # 内置音频试听功能
        def preview_builtin_audio(audio_name):
            """预览内置音频"""
            if not audio_name:
                return None

            # 使用共享的音频映射方法
            builtin_audio_map = self.get_builtin_audio_map()
            audio_path = builtin_audio_map.get(audio_name)
            if audio_path and os.path.exists(audio_path):
                return audio_path
            else:
                # 如果文件不存在，返回默认音频或None
                default_path = "builtin_audios/hunyin_6.mp3"
                if os.path.exists(default_path):
                    return default_path
                return None

        # 绑定模式选择变化事件
        self.mode_selection.change(
            fn=update_audio_components,
            inputs=[self.mode_selection],
            outputs=[self.builtin_audio_selection, self.builtin_audio_preview, self.reference_audio]
        )

        # 绑定内置音频选择变化事件，自动更新试听
        self.builtin_audio_selection.change(
            fn=preview_builtin_audio,
            inputs=[self.builtin_audio_selection],
            outputs=[self.builtin_audio_preview]
        )

        # 配置保存函数
        def save_current_config(*args):
            """保存当前配置"""
            try:
                config = {
                    "mode_selection": args[0],
                    "builtin_audio_selection": args[1],
                    "language": args[2],
                    "do_sample": args[3],
                    "temperature": args[4],
                    "top_p": args[5],
                    "top_k": args[6],
                    "num_beams": args[7],
                    "repetition_penalty": args[8],
                    "length_penalty": args[9],
                    "max_mel_tokens": args[10],
                    "volume_gain": args[11],
                    "max_text_tokens_per_sentence": args[12],
                    "sentences_bucket_max_size": args[13],
                    "infer_mode": args[14],
                    "api_url": args[15]
                }
                self._save_config(config)
            except Exception as e:
                logger.error(f"保存配置时出错: {e}")

        # 绑定配置保存事件 - 当任何参数改变时自动保存
        config_inputs = [
            self.mode_selection,
            self.builtin_audio_selection,
            self.language,
            self.do_sample,
            self.temperature,
            self.top_p,
            self.top_k,
            self.num_beams,
            self.repetition_penalty,
            self.length_penalty,
            self.max_mel_tokens,
            self.volume_gain,
            self.max_text_tokens_per_sentence,
            self.sentences_bucket_max_size,
            self.infer_mode,
            self.api_url
        ]

        # 为每个配置项绑定保存事件
        for component in config_inputs:
            component.change(
                fn=save_current_config,
                inputs=config_inputs,
                outputs=[]
            )

        # 添加标签切换时的配置恢复功能
        def restore_config_on_tab_switch():
            """标签切换时恢复配置"""
            try:
                current_config = self._load_config()
                logger.info(f"标签切换时恢复Index-TTS配置: {current_config}")

                # 返回所有组件的更新值
                return {
                    self.mode_selection: gr.update(value=current_config.get("mode_selection", "内置")),
                    self.builtin_audio_selection: gr.update(
                        value=current_config.get("builtin_audio_selection", "舒朗男声")),
                    self.language: gr.update(value=current_config.get("language", "中文")),
                    self.do_sample: gr.update(value=current_config.get("do_sample", True)),
                    self.temperature: gr.update(value=current_config.get("temperature", 1.0)),
                    self.top_p: gr.update(value=current_config.get("top_p", 0.8)),
                    self.top_k: gr.update(value=current_config.get("top_k", 30)),
                    self.volume_gain: gr.update(value=current_config.get("volume_gain", 1.0)),
                    self.api_url: gr.update(value=current_config.get("api_url", "http://127.0.0.1:7860"))
                }
            except Exception as e:
                logger.error(f"恢复配置时出错: {e}")
                return {}

        # 添加应用加载时的配置刷新机制
        def refresh_config_on_app_load():
            """应用加载时刷新配置"""
            try:
                fresh_config = self._load_config()

                # 更新组件可见性
                mode = fresh_config.get("mode_selection", "内置")
                return [
                    gr.update(value=mode),  # mode_selection
                    gr.update(
                        value=fresh_config.get("builtin_audio_selection", "舒朗男声"),
                        visible=(mode == "内置")
                    ),  # builtin_audio_selection
                    gr.update(visible=(mode == "内置")),  # builtin_audio_preview
                    gr.update(visible=(mode == "自定义")),  # reference_audio
                    gr.update(value=fresh_config.get("language", "中文")),  # language
                    gr.update(value=fresh_config.get("do_sample", True)),  # do_sample
                    gr.update(value=fresh_config.get("temperature", 1.0)),  # temperature
                    gr.update(value=fresh_config.get("top_p", 0.8)),  # top_p
                    gr.update(value=fresh_config.get("top_k", 30)),  # top_k
                    gr.update(value=fresh_config.get("volume_gain", 1.0)),  # volume_gain
                    gr.update(value=fresh_config.get("api_url", "http://127.0.0.1:7860"))  # api_url
                ]
            except Exception as e:
                logger.error(f"应用加载配置刷新失败: {e}")
                return [gr.update() for _ in range(11)]

        # 在应用加载时自动刷新配置
        # 使用Gradio的load事件
        app_load_outputs = [
            self.mode_selection, self.builtin_audio_selection, self.builtin_audio_preview,
            self.reference_audio, self.language, self.do_sample, self.temperature,
            self.top_p, self.top_k, self.volume_gain, self.api_url
        ]

        # 返回参数列表，按照其他TTS项目的格式
        INDEXTTS_ARGS = [
            self.reference_audio,
            self.mode_selection,
            self.builtin_audio_selection,
            self.language,
            self.do_sample,
            self.top_k,
            self.top_p,
            self.temperature,
            self.num_beams,
            self.repetition_penalty,
            self.length_penalty,
            self.max_mel_tokens,
            self.volume_gain,
            self.max_text_tokens_per_sentence,
            self.sentences_bucket_max_size,
            self.infer_mode,
            self.api_url
        ]

        # 存储配置刷新函数和输出组件，供外部调用
        self.refresh_config_on_app_load = refresh_config_on_app_load
        self.app_load_outputs = app_load_outputs

        return INDEXTTS_ARGS

    def before_gen_action(self, *args, **kwargs):
        """生成前的准备操作"""
        logger.info("准备生成Index-TTS语音...")
        return True

    def arg_filter(self, *args):
        """参数过滤，按照项目规范处理参数"""
        in_file, fps, offset, max_workers, reference_audio, mode_selection, builtin_audio_selection, language, do_sample, top_k, top_p, temperature, num_beams, repetition_penalty, length_penalty, max_mel_tokens, volume_gain, max_text_tokens_per_sentence, sentences_bucket_max_size, infer_mode, port = args

        # 根据模式验证必要参数
        if mode_selection == "自定义" and not reference_audio:
            raise Exception(i18n('Please upload reference audio for custom mode!'))
        elif mode_selection == "内置" and not builtin_audio_selection:
            raise Exception("请选择内置音频!")
        elif mode_selection == "clone":
            # Clone 模式预检查：确保视频已加载
            current_video_path = os.environ.get("current_video_path")
            if not current_video_path or not os.path.exists(current_video_path):
                raise Exception("🎬 Clone模式需要先加载视频文件！\n\n📁 请先点击右侧的'🚀 加载文件'按钮上传视频文件。")

            # 检查是否有 segments
            segments_found = self._check_segments_exist()
            if not segments_found:
                raise Exception("🎵 未找到音频分割片段！\n\n🔄 请确保视频文件已正确加载并完成音频分割处理。")

        pargs = (reference_audio, mode_selection, builtin_audio_selection, language, do_sample, top_k, top_p,
                 temperature, num_beams, repetition_penalty, length_penalty, max_mel_tokens, volume_gain,
                 max_text_tokens_per_sentence, sentences_bucket_max_size, infer_mode, port)
        kwargs = {'in_files': in_file, 'fps': fps, 'offset': offset, 'proj': "indextts", 'max_workers': max_workers}
        return pargs, kwargs
