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

    def api(self, port, text, reference_audio, language, do_sample, top_k, top_p, temperature,
            num_beams, repetition_penalty, length_penalty, max_mel_tokens,
            max_text_tokens_per_sentence, sentences_bucket_max_size, infer_mode):
        """调用Index-TTS API"""
        try:
            api_url = f'http://127.0.0.1:{port}'

            # 创建Gradio客户端
            client = Client(api_url, httpx_kwargs={"timeout": 7200, "proxy": None}, ssl_verify=False)

            # 检查参考音频文件是否存在
            if not reference_audio or not os.path.exists(reference_audio):
                logger.error(f"参考音频文件不存在: {reference_audio}")
                return None

            logger.info(f"Index-TTS API调用参数: text={text[:50]}..., infer_mode={infer_mode}, "
                       f"do_sample={do_sample}, top_k={top_k}, top_p={top_p}, temperature={temperature}, "
                       f"num_beams={num_beams}, repetition_penalty={repetition_penalty}, length_penalty={length_penalty}")

            # 调用Index-TTS的gen_single接口
            result = client.predict(
                handle_file(reference_audio),  # prompt
                text,                          # text
                infer_mode,                    # infer_mode
                int(max_text_tokens_per_sentence),  # max_text_tokens_per_sentence
                int(sentences_bucket_max_size),     # sentences_bucket_max_size
                do_sample,                     # do_sample
                float(top_p),                  # top_p
                int(top_k) if int(top_k) > 0 else 0,  # top_k
                float(temperature),            # temperature
                float(length_penalty),         # length_penalty
                int(num_beams),               # num_beams
                float(repetition_penalty),     # repetition_penalty
                int(max_mel_tokens),          # max_mel_tokens
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
        reference_audio, language, do_sample, top_k, top_p, temperature, num_beams, repetition_penalty, length_penalty, max_mel_tokens, max_text_tokens_per_sentence, sentences_bucket_max_size, infer_mode, port = args
        port = positive_int(port)

        audio = self.api(
            port=port,
            text=text,
            reference_audio=reference_audio,
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

                # 参考音频
                self.reference_audio = gr.Audio(label=i18n("Reference Audio"), type="filepath")

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

        # 返回参数列表，按照其他TTS项目的格式
        INDEXTTS_ARGS = [
            self.reference_audio,
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
        in_file, fps, offset, max_workers, reference_audio, language, do_sample, top_k, top_p, temperature, num_beams, repetition_penalty, length_penalty, max_mel_tokens, max_text_tokens_per_sentence, sentences_bucket_max_size, infer_mode, port = args

        # 验证必要参数
        if not reference_audio:
            raise Exception(i18n('Please upload reference audio!'))

        pargs = (reference_audio, language, do_sample, top_k, top_p, temperature, num_beams, repetition_penalty, length_penalty, max_mel_tokens, max_text_tokens_per_sentence, sentences_bucket_max_size, infer_mode, port)
        kwargs = {'in_files': in_file, 'fps': fps, 'offset': offset, 'proj': "indextts", 'max_workers': max_workers}
        return pargs, kwargs
