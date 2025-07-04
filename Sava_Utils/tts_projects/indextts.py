from . import TTSProjet
import gradio as gr
from .. import logger, i18n
from ..utils import positive_int
import requests
import os
from gradio_client import Client, handle_file

current_path = os.environ.get("current_path")


class IndexTTS(TTSProjet):
    def __init__(self, config):
        super().__init__("indextts", config)

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
            "清澈邻家弟弟": "builtin_audios/Chinese (Mandarin)_Pure-hearted_Boy.mp3"
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

    def api(self, port, text, reference_audio, mode_selection, builtin_audio_selection, language, do_sample, top_k,
            top_p, temperature,
            num_beams, repetition_penalty, length_penalty, max_mel_tokens,
            max_text_tokens_per_sentence, sentences_bucket_max_size, infer_mode):
        """调用Index-TTS API"""
        try:
            api_url = f'http://127.0.0.1:{port}'

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
                # clone模式的处理逻辑，这里可能需要特殊处理
                logger.info("使用clone模式，无需参考音频文件")
                audio_file_path = None  # clone模式可能不需要音频文件

            logger.info(f"Index-TTS API调用参数: text={text[:50]}..., mode={mode_selection}, "
                        f"builtin_audio={builtin_audio_selection if mode_selection == '内置' else 'N/A'}, "
                        f"language={language}, infer_mode={infer_mode}, do_sample={do_sample}, "
                        f"top_k={top_k}, top_p={top_p}, temperature={temperature}")

            # 调用Index-TTS的gen_single接口
            if audio_file_path:
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
            else:
                # clone模式或其他不需要音频文件的情况
                result = client.predict(
                    None,  # prompt (clone模式可能不需要)
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
        reference_audio, mode_selection, builtin_audio_selection, language, do_sample, top_k, top_p, temperature, num_beams, repetition_penalty, length_penalty, max_mel_tokens, max_text_tokens_per_sentence, sentences_bucket_max_size, infer_mode, port = args
        port = positive_int(port)

        audio = self.api(
            port=port,
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
        return audio

    def _UI(self):
        """创建Index-TTS的UI界面"""
        with gr.TabItem("Index-TTS"):
            with gr.Column():
                gr.Markdown("### Index-TTS 设置")

                # 参考音频模式 - 放在最上面
                self.mode_selection = gr.Radio(
                    label="参考音频模式",
                    choices=["内置", "clone", "自定义"],
                    value="内置",
                    interactive=True
                )

                # 内置音频选择 - 默认显示
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
                    value="舒朗男声",
                    visible=True,
                    interactive=True
                )

                # 内置音频试听 - 设置默认音频
                default_audio_path = self.get_default_builtin_audio()
                self.builtin_audio_preview = gr.Audio(
                    label="试听内置音频",
                    value=default_audio_path,
                    visible=True,
                    interactive=False
                )

                # 参考音频上传 - 初始隐藏
                self.reference_audio = gr.Audio(
                    label=i18n("Reference Audio"),
                    type="filepath",
                    visible=False
                )

                # 合成语言
                self.language = gr.Dropdown(
                    label="Synthesis Language",
                    choices=["中文", "英文", "日文", "中英混合", "日英混合", "中英日混合"],
                    value="中文",
                    interactive=True
                )

                with gr.Accordion("高级合成参数", open=False):
                    # 第一行 - 是否进行采样
                    with gr.Row():
                        self.do_sample = gr.Checkbox(
                            label="是否进行采样",
                            value=True,
                            interactive=True
                        )
                        self.temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            step=0.1,
                            value=1.0,
                            label="temperature"
                        )

                    # 第二行 - top_p, top_k, num_beams
                    with gr.Row():
                        self.top_p = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=0.8,
                            label="top_p"
                        )
                        self.top_k = gr.Slider(
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=30,
                            label="top_k"
                        )
                        self.num_beams = gr.Slider(
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=3,
                            label="num_beams"
                        )

                    # 第三行 - repetition_penalty 和 length_penalty
                    with gr.Row():
                        self.repetition_penalty = gr.Number(
                            label="repetition_penalty",
                            value=10,
                            precision=1,
                            interactive=True
                        )
                        self.length_penalty = gr.Number(
                            label="length_penalty",
                            value=0,
                            precision=1,
                            interactive=True
                        )

                    # 第四行 - max_mel_tokens
                    with gr.Row():
                        self.max_mel_tokens = gr.Slider(
                            minimum=50,
                            maximum=800,
                            step=10,
                            value=600,
                            label="max_mel_tokens",
                            info="生成Token最大数量，过小导致音频被截断"
                        )

                    # 分句设置 - 参数会影响音频质量和生成速度
                    gr.Markdown("### 分句设置 *参数会影响音频质量和生成速度*")
                    with gr.Row():
                        self.max_text_tokens_per_sentence = gr.Slider(
                            minimum=20,
                            maximum=600,
                            step=10,
                            value=120,
                            label="分句最大Token数",
                            info="建议20~300之间，值越大，单次分句越长，过小大幅增加推理次数导致速度最慢"
                        )
                        self.sentences_bucket_max_size = gr.Slider(
                            minimum=1,
                            maximum=16,
                            step=1,
                            value=4,
                            label="分句分桶最大容量(批次推理主效)",
                            info="建议2~8之间，值越大，一批次推理的分句数量越多，过大可能导致显存溢出"
                        )

                    # 推理模式
                    with gr.Row():
                        self.infer_mode = gr.Radio(
                            label="Inference Mode",
                            choices=["普通推理", "批次推理"],
                            value="普通推理",
                            interactive=True
                        )

                # API端口
                self.api_port5 = gr.Number(
                    label="API Port",
                    value=7860,
                    precision=0,
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
            self.max_text_tokens_per_sentence,
            self.sentences_bucket_max_size,
            self.infer_mode,
            self.api_port5
        ]
        return INDEXTTS_ARGS

    def before_gen_action(self, *args, **kwargs):
        """生成前的准备操作"""
        logger.info("准备生成Index-TTS语音...")
        return True

    def arg_filter(self, *args):
        """参数过滤，按照项目规范处理参数"""
        in_file, fps, offset, max_workers, reference_audio, mode_selection, builtin_audio_selection, language, do_sample, top_k, top_p, temperature, num_beams, repetition_penalty, length_penalty, max_mel_tokens, max_text_tokens_per_sentence, sentences_bucket_max_size, infer_mode, port = args

        # 根据模式验证必要参数
        if mode_selection == "自定义" and not reference_audio:
            raise Exception(i18n('Please upload reference audio for custom mode!'))
        elif mode_selection == "内置" and not builtin_audio_selection:
            raise Exception("请选择内置音频!")

        pargs = (reference_audio, mode_selection, builtin_audio_selection, language, do_sample, top_k, top_p,
                 temperature, num_beams, repetition_penalty, length_penalty, max_mel_tokens,
                 max_text_tokens_per_sentence, sentences_bucket_max_size, infer_mode, port)
        kwargs = {'in_files': in_file, 'fps': fps, 'offset': offset, 'proj': "indextts", 'max_workers': max_workers}
        return pargs, kwargs
