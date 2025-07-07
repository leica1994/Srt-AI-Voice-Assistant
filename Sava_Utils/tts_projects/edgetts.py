import os
import asyncio
import gradio as gr
import edge_tts
import ffmpeg
from pydub import AudioSegment

from . import TTSProjet
from .. import logger

current_path = os.environ.get("current_path", os.getcwd())


class EdgeTTS(TTSProjet):
    def __init__(self, config):
        self.edge_voices = {}
        self.voice_list = []
        self.language_map = self._get_language_map()
        self.voice_name_map = self._get_voice_name_map()
        super().__init__("edgetts", config)
        self.load_voices()

    def _get_language_map(self):
        """语言代码映射（保持原始显示）"""
        return {}

    def _get_voice_name_map(self):
        """语音名称映射（保持原始显示）"""
        return {}

    def load_voices(self):
        """加载 Edge-TTS 可用的语音列表"""
        if edge_tts is None:
            logger.error("edge-tts 库未安装，请运行: pip install edge-tts")
            self.edge_voices = {}
            self.voice_list = []
            return

        try:
            # 异步获取语音列表
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            voices = loop.run_until_complete(edge_tts.list_voices())
            loop.close()

            # 初始化数据结构
            self.edge_voices = {}

            # 处理每个语音
            for voice in voices:
                locale = voice['Locale']
                friendly_name = voice['FriendlyName']

                # 使用原始语言代码作为分组
                if locale not in self.edge_voices:
                    self.edge_voices[locale] = {}

                # 使用原始语音名称
                self.edge_voices[locale][friendly_name] = {
                    'ShortName': voice['ShortName'],
                    'Gender': voice['Gender'],
                    'Locale': voice['Locale'],
                    'SuggestedCodec': voice.get('SuggestedCodec', 'audio-24khz-48kbitrate-mono-mp3'),
                    'FriendlyName': voice['FriendlyName']
                }

            # 生成语音列表，优先显示中文相关语音
            chinese_languages = [lang for lang in self.edge_voices.keys() if 'zh-' in lang.lower()]
            other_languages = [lang for lang in self.edge_voices.keys() if 'zh-' not in lang.lower()]
            self.voice_list = sorted(chinese_languages) + sorted(other_languages)

            logger.info(f"成功加载 {len(voices)} 个 Edge-TTS 语音")

        except Exception as e:
            logger.error(f"加载 Edge-TTS 语音列表失败: {e}")
            self.edge_voices = {}
            self.voice_list = []

    async def _generate_speech_async(self, text, voice, rate, pitch):
        """异步生成语音

        Args:
            text (str): 要合成的文本
            voice (str): 语音名称
            rate (float): 语速倍率
            pitch (float): 音调倍率

        Returns:
            bytes: 音频数据，如果失败返回 None
        """
        try:
            # 构建语速和音调参数
            rate_str = f"{int((rate - 1) * 100):+d}%"
            pitch_str = f"{int((pitch - 1) * 50):+d}Hz"

            # 创建 TTS 通信对象
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=rate_str,
                pitch=pitch_str
            )

            # 收集音频数据
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            # 验证和处理音频数据
            return self._process_audio_data(audio_data)

        except Exception as e:
            logger.error(f"异步语音生成失败: {e}")
            return None

    def _process_audio_data(self, audio_data):
        """处理音频数据，确保格式正确"""
        if not audio_data or len(audio_data) <= 44:
            logger.error("音频数据无效或过小")
            return None

        # 检查是否是标准WAV格式
        if audio_data[:4] == b'RIFF' and audio_data[8:12] == b'WAVE':
            return audio_data
        else:
            # 尝试转换为WAV格式
            logger.warning("检测到非标准WAV格式，尝试转换...")
            return self._convert_to_wav(audio_data)

    def api(self, language, speaker, rate, pitch, text, **kwargs):
        """调用 Edge-TTS API 生成语音

        Args:
            language (str): 语言名称
            speaker (str): 语音名称
            rate (float): 语速倍率
            pitch (float): 音调倍率
            text (str): 要合成的文本

        Returns:
            bytes: 音频数据，如果失败返回 None
        """
        if edge_tts is None:
            logger.error("edge-tts 库未安装")
            return None

        try:
            # 验证语音参数
            if not self._validate_voice_params(language, speaker):
                return None

            # 获取语音信息
            voice_info = self.edge_voices[language][speaker]
            voice_short_name = voice_info['ShortName']

            # 异步生成语音
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                audio_data = loop.run_until_complete(
                    self._generate_speech_async(text, voice_short_name, rate, pitch)
                )
            finally:
                loop.close()

            # 验证结果
            if audio_data:
                logger.info(f"成功生成语音，大小: {len(audio_data)} 字节")
                return audio_data
            else:
                logger.error("生成的音频数据为空")
                return None

        except Exception as e:
            logger.error(f"Edge-TTS API 调用失败: {e}")
            return None

    def _validate_voice_params(self, language, speaker):
        """验证语音参数"""
        if language not in self.edge_voices:
            logger.error(f"未找到语言: {language}")
            return False

        if speaker not in self.edge_voices[language]:
            logger.error(f"未找到语音: {language} - {speaker}")
            return False

        return True

    def _convert_to_wav(self, audio_data):
        """将MP3音频数据转换为WAV格式"""
        try:
            import io
            import tempfile
            import os

            # 检查是否是MP3格式
            if audio_data[:2] == b'\xff\xf3' or audio_data[:3] == b'ID3':
                logger.info("检测到MP3格式，转换为WAV...")

                # 使用pydub转换MP3到WAV
                try:
                    # 从字节数据加载MP3
                    audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))

                    # 转换为WAV格式
                    wav_buffer = io.BytesIO()
                    audio_segment.export(wav_buffer, format="wav")
                    wav_data = wav_buffer.getvalue()

                    logger.info(f"MP3转WAV成功，大小: {len(wav_data)} 字节")
                    return wav_data

                except ImportError:
                    logger.warning("pydub未安装，使用临时文件方式转换...")

                    # 使用临时文件和ffmpeg转换
                    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as mp3_file:
                        mp3_file.write(audio_data)
                        mp3_path = mp3_file.name

                    wav_path = mp3_path.replace('.mp3', '.wav')

                    # 使用 ffmpeg-python 转换
                    try:
                        (
                            ffmpeg
                            .input(mp3_path)
                            .output(wav_path, acodec='pcm_s16le', ar=24000, ac=1)
                            .overwrite_output()
                            .run(quiet=True)
                        )

                        if os.path.exists(wav_path):
                            with open(wav_path, 'rb') as wav_file:
                                wav_data = wav_file.read()

                            # 清理临时文件
                            os.unlink(mp3_path)
                            os.unlink(wav_path)

                            logger.info(f"ffmpeg-python转换成功，大小: {len(wav_data)} 字节")
                            return wav_data
                        else:
                            logger.error("ffmpeg-python转换失败：输出文件不存在")
                            # 清理临时文件
                            if os.path.exists(mp3_path):
                                os.unlink(mp3_path)
                    except ffmpeg.Error as e:
                        logger.error(f"ffmpeg-python转换失败: {e}")
                        # 清理临时文件
                        if os.path.exists(mp3_path):
                            os.unlink(mp3_path)
                        if os.path.exists(wav_path):
                            os.unlink(wav_path)
                        return None
            else:
                logger.warning("未知音频格式，尝试直接返回")
                return audio_data

        except Exception as e:
            logger.error(f"音频格式转换失败: {e}")
            return None

    def _UI(self):
        """创建 Edge-TTS 的 UI 界面"""
        with gr.TabItem("🎤 Edge-TTS"):
            with gr.Column():

                if not self.edge_voices:
                    gr.Markdown("⚠️ **Edge-TTS 不可用**")
                    gr.Markdown("请安装 edge-tts 库: `pip install edge-tts`")

                    # 创建空的组件
                    self.edge_language = gr.Dropdown(
                        label="语言",
                        choices=[],
                        value=None,
                        interactive=False
                    )
                    self.edge_speaker = gr.Dropdown(
                        label="语音",
                        choices=[],
                        value=None,
                        interactive=False
                    )
                    self.edge_rate = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        step=0.1,
                        value=1.0,
                        label="语速",
                        interactive=False
                    )
                    self.edge_pitch = gr.Slider(
                        minimum=0.5,
                        maximum=1.5,
                        step=0.1,
                        value=1.0,
                        label="音调",
                        interactive=False
                    )
                    self.gen_btn_edge = gr.Button(
                        value="生成音频",
                        variant="primary",
                        interactive=False
                    )
                else:
                    # 语言选择
                    language_choices = list(self.edge_voices.keys())
                    # 优先选择 zh-CN，然后是其他中文语言，最后是其他语言
                    if 'zh-CN' in language_choices:
                        default_language = 'zh-CN'
                    else:
                        default_language = next((lang for lang in language_choices if 'zh' in lang.lower()),
                                                language_choices[0] if language_choices else None)

                    self.edge_language = gr.Dropdown(
                        label="🌍 选择语言",
                        choices=language_choices,
                        value=default_language,
                        interactive=True,
                        info="选择语音合成的语言"
                    )

                    # 语音选择
                    if default_language and default_language in self.edge_voices:
                        speaker_choices = list(self.edge_voices[default_language].keys())
                        default_speaker = speaker_choices[0] if speaker_choices else None
                    else:
                        speaker_choices = []
                        default_speaker = None

                    self.edge_speaker = gr.Dropdown(
                        label="🎭 选择语音",
                        choices=speaker_choices,
                        value=default_speaker,
                        interactive=True,
                        info="选择语音角色"
                    )

                    # 语音试听区域 - 类似 Index-TTS 的样式
                    # 生成初始试听音频
                    initial_preview = None
                    if default_language and default_speaker:
                        initial_preview = self.auto_preview_voice(default_language, default_speaker, 1.0, 1.0)

                    self.preview_audio = gr.Audio(
                        label="🎵 语音试听",
                        value=initial_preview,
                        visible=True,
                        interactive=False,
                        show_label=True,
                        container=True,
                        show_download_button=True
                    )

                    # 语音参数控制
                    with gr.Group():
                        gr.Markdown("#### 🎛️ 语音参数")
                        with gr.Row():
                            self.edge_rate = gr.Slider(
                                minimum=0.5,
                                maximum=2.0,
                                step=0.1,
                                value=1.0,
                                label="语速",
                                info="控制语音播放速度"
                            )
                            self.edge_pitch = gr.Slider(
                                minimum=0.5,
                                maximum=1.5,
                                step=0.1,
                                value=1.0,
                                label="音调",
                                info="控制语音音调高低"
                            )

                    # 生成按钮
                    self.gen_btn_edge = gr.Button(
                        value="🎵 生成音频",
                        variant="primary"
                    )

                    # 绑定事件
                    self.edge_language.change(
                        fn=self.update_speakers,
                        inputs=[self.edge_language],
                        outputs=[self.edge_speaker]
                    )

                    # 绑定语音选择改变事件，自动加载试听音频
                    self.edge_speaker.change(
                        fn=self.auto_preview_voice,
                        inputs=[self.edge_language, self.edge_speaker, self.edge_rate, self.edge_pitch],
                        outputs=[self.preview_audio]
                    )

                    # 绑定参数改变事件，自动更新试听
                    self.edge_rate.change(
                        fn=self.auto_preview_voice,
                        inputs=[self.edge_language, self.edge_speaker, self.edge_rate, self.edge_pitch],
                        outputs=[self.preview_audio]
                    )

                    self.edge_pitch.change(
                        fn=self.auto_preview_voice,
                        inputs=[self.edge_language, self.edge_speaker, self.edge_rate, self.edge_pitch],
                        outputs=[self.preview_audio]
                    )

                # 返回参数列表
                EDGETTS_ARGS = [
                    self.edge_language,
                    self.edge_speaker,
                    self.edge_rate,
                    self.edge_pitch
                ]

        return EDGETTS_ARGS

    def update_speakers(self, language):
        """更新语音选择列表"""
        if not language or language not in self.edge_voices:
            return gr.update(choices=[], value=None)

        choices = list(self.edge_voices[language].keys())
        return gr.update(choices=choices, value=choices[0] if choices else None)

    def auto_preview_voice(self, language, speaker, rate, pitch):
        """自动试听语音 - 类似 Index-TTS 的切换即加载"""
        if not language or not speaker or language not in self.edge_voices or speaker not in self.edge_voices[language]:
            return None

        try:
            # 使用示例文本进行试听（基于语言代码）
            preview_texts = {
                'zh-CN': "你好，我是{speaker}，这是语音试听。",
                'zh-HK': "你好，我係{speaker}，呢個係語音試聽。",
                'zh-TW': "你好，我是{speaker}，這是語音試聽。",
                'en-US': "Hello, I'm {speaker}. This is a voice preview.",
                'en-GB': "Hello, I'm {speaker}. This is a voice preview.",
                'en-AU': "Hello, I'm {speaker}. This is a voice preview.",
                'ja-JP': "こんにちは、私は{speaker}です。これは音声プレビューです。",
                'ko-KR': "안녕하세요, 저는 {speaker}입니다. 이것은 음성 미리보기입니다。",
                'fr-FR': "Bonjour, je suis {speaker}. Ceci est un aperçu vocal.",
                'de-DE': "Hallo, ich bin {speaker}. Das ist eine Sprachvorschau.",
                'es-ES': "Hola, soy {speaker}. Esta es una vista previa de voz.",
                'it-IT': "Ciao, sono {speaker}. Questa è un'anteprima vocale.",
                'pt-BR': "Olá, eu sou {speaker}. Esta é uma prévia de voz.",
                'ru-RU': "Привет, я {speaker}. Это предварительный просмотр голоса.",
                'ar-SA': "مرحبا، أنا {speaker}. هذه معاينة صوتية.",
                'hi-IN': "नमस्ते, मैं {speaker} हूँ। यह एक आवाज़ पूर्वावलोकन है।",
                'th-TH': "สวัสดี ฉันคือ {speaker} นี่คือการแสดงตัวอย่างเสียง",
                'vi-VN': "Xin chào, tôi là {speaker}. Đây là bản xem trước giọng nói."
            }

            # 根据语言选择试听文本
            preview_text = preview_texts.get(language, "Hello, this is a voice preview.")
            # 替换语音名称占位符
            preview_text = preview_text.replace("{speaker}", speaker.split('（')[0])

            # 生成试听音频
            audio_data = self.api(language, speaker, rate, pitch, preview_text)
            if audio_data:
                # 保存临时音频文件
                import tempfile

                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    # Edge-TTS 返回的是 MP3 格式，需要转换
                    if audio_data[:2] == b'\xff\xf3' or audio_data[:3] == b'ID3':
                        # 转换 MP3 到 WAV
                        converted_audio = self._convert_to_wav(audio_data)
                        if converted_audio:
                            temp_file.write(converted_audio)
                        else:
                            temp_file.write(audio_data)  # 如果转换失败，使用原始数据
                    else:
                        temp_file.write(audio_data)

                    temp_file.flush()
                    logger.info(f"自动试听成功: {speaker}")
                    return temp_file.name
            else:
                logger.warning("自动试听失败")
                return None

        except Exception as e:
            logger.error(f"自动试听出错: {e}")
            return None

    def save_action(self, *args, text: str = None):
        """保存操作，调用API并返回音频数据"""
        language, speaker, rate, pitch = args
        audio = self.api(language, speaker, rate, pitch, text)
        return audio

    def before_gen_action(self, *args, **kwargs):
        """生成前的准备操作"""
        if edge_tts is None:
            raise Exception("edge-tts 库未安装，请运行: pip install edge-tts")
        logger.info("准备生成 Edge-TTS 语音...")

    def arg_filter(self, *args):
        """参数过滤，按照项目规范处理参数"""
        input_file, fps, offset, workers, language, speaker, rate, pitch = args

        if not speaker:
            raise Exception("请选择语音!")

        if edge_tts is None:
            raise Exception("edge-tts 库未安装，请运行: pip install edge-tts")

        pargs = (language, speaker, rate, pitch)
        kwargs = {
            'in_files': input_file,
            'fps': fps,
            'offset': offset,
            'proj': "edgetts",
            'max_workers': workers
        }
        return pargs, kwargs
