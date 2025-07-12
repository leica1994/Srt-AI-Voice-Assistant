import os
import json
import requests
import base64
import tempfile
import gradio as gr
from . import TTSProjet
from .. import logger, i18n

current_path = os.environ.get("current_path")

try:
    dict_language: dict = i18n('DICT_LANGUAGE')
    assert type(dict_language) is dict
except:
    dict_language = {
        "Chinese": "all_zh",
        "Cantonese": "all_yue", 
        "English": "en",
        "Japanese": "all_ja",
        "Korean": "all_ko",
        "Chinese-English Mix": "zh",
        "Cantonese-English Mix": "yue",
        "Japanese-English Mix": "ja",
        "Korean-English Mix": "ko",
        "Multi-Language Mix": "auto",
        "Multi-Language Mix (Cantonese)": "auto_yue",
    }


def temp_ra(a):
    """处理参考音频，支持多种输入格式"""
    import tempfile
    
    try:
        # 如果是元组格式 (sr, wav)
        if isinstance(a, tuple) and len(a) == 2:
            sr, wav = a
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                import soundfile as sf
                sf.write(tmp_file.name, wav, sr)
                return tmp_file.name
        # 如果是文件路径字符串
        elif isinstance(a, str):
            return a
        # 如果是 Gradio Audio 组件的输出
        elif hasattr(a, 'name'):
            return a.name
        else:
            # 尝试直接返回，可能是文件路径
            return str(a)
    except Exception as e:
        logger.error(f"处理参考音频失败: {e}, 输入类型: {type(a)}, 输入内容: {a}")
        return ""


class CosyVoice2(TTSProjet):
    def __init__(self, config):
        self.config_file = os.path.join(current_path, "outputs", "cosyvoice2_config.json")
        super().__init__("cosyvoice2", config)

    def _save_config(self, config):
        """保存配置"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存CosyVoice2配置失败: {e}")

    def _load_config(self):
        """加载配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"加载CosyVoice2配置失败: {e}")
        return self._get_default_config()

    def _get_default_config(self):
        """获取默认配置"""
        return {
            "language": "zh",
            "api_url": "http://127.0.0.1:9881",
            "refer_text": "",
            "refer_lang": "auto_yue",
            "inference_mode": "3s极速复刻"
        }

    def update_cfg(self, config):
        self.server_mode = config.server_mode

    def api(self, **kwargs):
        try:
            # 参考 gsv.py 中的 CosyVoice2 实现
            import io
            import wave

            api_url = kwargs.get("api_url", "http://127.0.0.1:9881").strip().rstrip('/')
            files = None

            logger.info("使用3s克隆模式...")
            data_json = {
                "prompt_text": kwargs["prompt_text"],
                "tts_text": kwargs["text"],
                "speed": kwargs.get("speed_factor", 1.0)
            }
            API_URL = f"{api_url}/inference_zero_shot"
            files = [('prompt_wav', ('prompt_wav', open(kwargs["ref_audio_path"], 'rb'), 'application/octet-stream'))]

            logger.info(f"🔗 CosyVoice2 API URL: {API_URL}")
            logger.debug(f"📋 请求数据: {data_json}")

            response = requests.request("GET", url=API_URL, data=data_json, files=files, stream=False)
            response.raise_for_status()

            # 将响应内容转换为 WAV 格式
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)  # cosy api does not provide sr.
                wav_file.writeframes(response.content)

            logger.info(f"✅ CosyVoice2 API 调用成功，音频大小: {len(wav_buffer.getvalue())} 字节")
            return wav_buffer.getvalue()

        except Exception as e:
            err = f"CosyVoice2 API调用失败: {e}"
            try:
                err += f" 返回消息: {response.json()}"
            except:
                pass
            logger.error(err)
            raise Exception(err)

    def _UI(self):
        # 加载保存的配置
        saved_config = self._load_config()
        
        with gr.TabItem("🎵 CosyVoice2"):
            
            self.language = gr.Dropdown(
                choices=list(dict_language.items()), 
                value=saved_config.get("language", "zh"), 
                label=i18n('Inference text language'), 
                interactive=True, 
                allow_custom_value=False
            )
            
            with gr.Accordion("参考音频设置", open=True):
                gr.Markdown("💡 **使用说明**: 上传参考音频文件启用3s极速复刻模式，不上传则使用预训练音色模式")
                self.refer_audio = gr.Audio(label="参考音频 (可选)")
                with gr.Row():
                    self.refer_text = gr.Textbox(
                        label="参考文本",
                        value=saved_config.get("refer_text", ""),
                        placeholder="3s极速复刻: 参考音频的转录文本 | 预训练音色: 说话人ID"
                    )
                    self.refer_lang = gr.Dropdown(
                        choices=list(dict_language.items()),
                        value=saved_config.get("refer_lang", "auto_yue"),
                        label="参考音频语言",
                        interactive=True,
                        allow_custom_value=False
                    )
            
            with gr.Row():
                self.api_url = gr.Textbox(
                    label="CosyVoice2 API URL (默认端口 9881)",
                    value=saved_config.get("api_url", "http://127.0.0.1:9881"),
                    interactive=not self.server_mode,
                    visible=not self.server_mode
                )

            with gr.Row():
                self.gen_btn = gr.Button(
                    value=i18n('Generate Audio'), 
                    variant="primary", 
                    visible=True
                )
        
        COSYVOICE2_ARGS = [
            self.language,
            self.api_url,
            self.refer_audio,
            self.refer_text,
            self.refer_lang,
        ]

        # 添加参数记忆功能 - 当任何参数改变时保存配置
        config_inputs = [
            self.language,
            self.api_url,
            self.refer_text,
            self.refer_lang
        ]
        
        # 为每个配置项绑定保存事件
        for component in config_inputs:
            component.change(
                fn=lambda *args: self._save_current_config(*args),
                inputs=config_inputs,
                outputs=[]
            )
        
        return COSYVOICE2_ARGS
    
    def _save_current_config(self, *args):
        """保存当前配置"""
        try:
            config = {
                "language": args[0],
                "api_url": args[1],
                "refer_text": args[2],
                "refer_lang": args[3]
            }
            self._save_config(config)
        except Exception as e:
            logger.error(f"保存CosyVoice2配置时出错: {e}")

    def arg_filter(self, *args):
        in_file, fps, offset, max_workers, language, api_url, refer_audio, refer_text, refer_lang = args

        # CosyVoice2 不强制要求参考音频，可以使用预训练音色
        if refer_audio is not None:
            refer_audio_path = temp_ra(refer_audio)
        else:
            refer_audio_path = ''

        pargs = (
            "CosyVoice2",
            dict_language.get(language, language),
            api_url,
            refer_audio_path,
            refer_text,
            dict_language.get(refer_lang, refer_lang)
        )
        kwargs = {'in_files': in_file, 'fps': fps, 'offset': offset, 'proj': "cosyvoice2", 'max_workers': max_workers}
        return pargs, kwargs

    def before_gen_action(self, *args, **kwargs):
        # CosyVoice2 不需要模型切换，直接返回
        pass

    def save_action(self, *args, text: str = None):
        """保存操作，调用API并返回音频数据"""
        try:
            # 调试：打印参数数量和内容
            logger.debug(f"CosyVoice2 save_action 收到参数数量: {len(args)}")
            logger.debug(f"CosyVoice2 save_action 参数内容: {args}")

            # 根据实际参数数量进行解包
            if len(args) >= 6:
                project_name = args[0]  # "CosyVoice2"
                language = args[1]
                api_url = args[2]
                refer_audio_path = args[3]  # 已经是路径字符串
                refer_text = args[4]
                refer_lang = args[5]
            else:
                # 如果参数数量不匹配，使用默认值
                project_name = args[0] if len(args) > 0 else "CosyVoice2"
                language = args[1] if len(args) > 1 else "zh"
                api_url = args[2] if len(args) > 2 else "http://127.0.0.1:9881"
                refer_audio_path = args[3] if len(args) > 3 else ""
                refer_text = args[4] if len(args) > 4 else ""
                refer_lang = args[5] if len(args) > 5 else "auto_yue"

            # refer_audio_path 已经是处理过的路径字符串
            ref_audio_path = refer_audio_path if refer_audio_path else ''

            # 调试信息
            logger.debug(f"CosyVoice2 API调用参数:")
            logger.debug(f"  language: {language}")
            logger.debug(f"  api_url: {api_url}")
            logger.debug(f"  refer_text: {refer_text}")
            logger.debug(f"  refer_lang: {refer_lang}")
            logger.debug(f"  text: {text}")

            # 调用API
            audio = self.api(
                text_lang=dict_language.get(language, language),
                api_url=api_url,
                ref_audio_path=ref_audio_path,
                prompt_text=refer_text,
                prompt_lang=dict_language.get(refer_lang, refer_lang),
                text=text
            )
            return audio
        except Exception as e:
            logger.error(f"CosyVoice2 API调用失败: {e}")
            return None
