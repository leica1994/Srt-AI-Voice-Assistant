import os
import json
import time
import shutil
import requests
import soundfile as sf
import gradio as gr
from . import TTSProjet
from .. import logger, i18n

current_path = os.environ.get("current_path")

try:
    dict_language: dict = i18n('DICT_LANGUAGE')
    assert type(dict_language) is dict
    cut_method: dict = i18n('CUT_METHOD')
    assert type(cut_method) is dict
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
    cut_method = {
        "Slice per 4 sentences": "cut0",
        "Slice per 50 characters": "cut1",
        "Slice per 50 characters": "cut2",
        "Slice by Chinese punct": "cut3",
        "Slice by English punct": "cut4",
        "Slice by every punct": "cut5",
    }


def temp_ra(a: tuple):
    sr, wav = a
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        sf.write(tmp_file.name, wav, sr)
        return tmp_file.name


def temp_aux_ra(a):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_file.write(a)
        return tmp_file.name


class GPTSoVITS(TTSProjet):
    def __init__(self, config):
        self.current_sovits_model = dict()
        self.current_gpt_model = dict()
        self.config_file = os.path.join(current_path, "outputs", "gpt_sovits_config.json")
        super().__init__("gpt_sovits", config)

    def _save_config(self, config):
        """保存配置"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存GPT-SoVITS配置失败: {e}")

    def _load_config(self):
        """加载配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"加载GPT-SoVITS配置失败: {e}")
        return self._get_default_config()

    def _get_default_config(self):
        """获取默认配置"""
        return {
            "language": "zh",
            "api_url": "http://127.0.0.1:9880",
            "refer_text": "",
            "refer_lang": "auto_yue",
            "top_k": 5,
            "top_p": 1.0,
            "temperature": 1.0,
            "text_split_method": "cut5",
            "batch_size": 20,
            "batch_threshold": 0.75,
            "split_bucket": True,
            "speed_factor": 1.0,
            "fragment_interval": 0.3,
            "seed": -1,
            "parallel_infer": True,
            "repetition_penalty": 1.35,
            "sample_steps": "32",
            "how_to_cut": "cut5"
        }

    def update_cfg(self, config):
        self.server_mode = config.server_mode

    def api(self, **kwargs):
        try:
            if kwargs["ref_audio_path"] == '':
                data_json = {
                    "refer_wav_path": "",
                    "prompt_text": kwargs["prompt_text"],
                    "prompt_language": kwargs["prompt_lang"],
                    "text": kwargs["text"],
                    "text_language": kwargs["text_lang"],
                    "cut_punc": kwargs["text_split_method"],
                    "top_k": kwargs["top_k"],
                    "top_p": kwargs["top_p"],
                    "temperature": kwargs["temperature"],
                    "speed": kwargs["speed_factor"],
                    "inp_refs": kwargs["aux_ref_audio_paths"],
                    "sample_steps": kwargs["sample_steps"],
                }
                API_URL = f"{kwargs['api_url']}/"
            else:
                data_json = {
                    "refer_wav_path": kwargs["ref_audio_path"],
                    "prompt_text": kwargs["prompt_text"],
                    "prompt_language": kwargs["prompt_lang"],
                    "text": kwargs["text"],
                    "text_language": kwargs["text_lang"],
                    "cut_punc": kwargs["text_split_method"],
                    "top_k": kwargs["top_k"],
                    "top_p": kwargs["top_p"],
                    "temperature": kwargs["temperature"],
                    "speed": kwargs["speed_factor"],
                    "inp_refs": kwargs["aux_ref_audio_paths"],
                    "sample_steps": kwargs["sample_steps"],
                }
                API_URL = f"{kwargs['api_url']}/"
            
            response = requests.post(url=API_URL, json=data_json)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"GPT-SoVITS API调用失败: {e}")
            raise e

    def _UI(self):
        # 加载保存的配置
        saved_config = self._load_config()
        
        with gr.TabItem("🎯 GPT-SoVITS"):
            self.language = gr.Dropdown(
                choices=list(dict_language.items()), 
                value=saved_config.get("language", "zh"), 
                label=i18n('Inference text language'), 
                interactive=True, 
                allow_custom_value=False
            )
            
            with gr.Accordion(i18n('Reference Audio'), open=True):                
                self.refer_audio = gr.Audio(label=i18n('Main Reference Audio'))
                self.aux_ref_audio = gr.File(
                    label=i18n('Auxiliary Reference Audios'), 
                    file_types=['.wav'], 
                    file_count="multiple", 
                    type="binary"
                )
                with gr.Row():
                    self.refer_text = gr.Textbox(
                        label=i18n('Transcription of Main Reference Audio'), 
                        value=saved_config.get("refer_text", ""), 
                        placeholder=i18n('Transcription')
                    )
                    self.refer_lang = gr.Dropdown(
                        choices=list(dict_language.items()), 
                        value=saved_config.get("refer_lang", "auto_yue"), 
                        label=i18n('Language of Main Reference Audio'), 
                        interactive=True, 
                        allow_custom_value=False
                    )
            
            with gr.Accordion(i18n('Switch Models'), open=False, visible=not self.server_mode):
                self.sovits_path = gr.Dropdown(
                    value="", 
                    label=f"Sovits {i18n('Model Path')}", 
                    interactive=True, 
                    allow_custom_value=True, 
                    choices=['']
                )
                self.gpt_path = gr.Dropdown(
                    value="", 
                    label=f"GPT {i18n('Model Path')}", 
                    interactive=True, 
                    allow_custom_value=True, 
                    choices=['']
                )
                with gr.Row():                
                    self.switch_model_btn = gr.Button(
                        value=i18n('Switch Models'), 
                        variant="primary", 
                        scale=4
                    )
                    self.scan_model_btn = gr.Button(
                        value=i18n('🔄️'), 
                        variant="secondary", 
                        scale=1, 
                        min_width=60
                    )
                    self.scan_model_btn.click(
                        self.find_models,
                        inputs=[],
                        outputs=[self.sovits_path, self.gpt_path]
                    )
            
            with gr.Row():
                self.api_url = gr.Textbox(
                    label="API URL", 
                    value=saved_config.get("api_url", "http://127.0.0.1:9880"), 
                    interactive=not self.server_mode, 
                    visible=not self.server_mode
                )
            
            with gr.Accordion(i18n('Advanced Parameters'), open=False):
                self.batch_size = gr.Slider(
                    minimum=1, maximum=200, step=1, 
                    label="batch_size", 
                    value=saved_config.get("batch_size", 20), 
                    interactive=True
                )
                self.batch_threshold = gr.Slider(
                    minimum=0, maximum=1, step=0.01, 
                    label="batch_threshold", 
                    value=saved_config.get("batch_threshold", 0.75), 
                    interactive=True
                )
                self.fragment_interval = gr.Slider(
                    minimum=0.01, maximum=1, step=0.01, 
                    label=i18n('Fragment Interval(sec)'), 
                    value=saved_config.get("fragment_interval", 0.3), 
                    interactive=True
                )
                self.speed_factor = gr.Slider(
                    minimum=0.25, maximum=4, step=0.05, 
                    label="speed_factor", 
                    value=saved_config.get("speed_factor", 1.0), 
                    interactive=True
                )
                self.top_k = gr.Slider(
                    minimum=1, maximum=100, step=1, 
                    label="top_k", 
                    value=saved_config.get("top_k", 5), 
                    interactive=True
                )
                self.top_p = gr.Slider(
                    minimum=0, maximum=1, step=0.05, 
                    label="top_p", 
                    value=saved_config.get("top_p", 1.0), 
                    interactive=True
                )
                self.temperature = gr.Slider(
                    minimum=0, maximum=1, step=0.05, 
                    label="temperature", 
                    value=saved_config.get("temperature", 1.0), 
                    interactive=True
                )
                self.repetition_penalty = gr.Slider(
                    minimum=0, maximum=2, step=0.05, 
                    label="repetition_penalty", 
                    value=saved_config.get("repetition_penalty", 1.35), 
                    interactive=True
                )
                self.sample_steps = gr.Dropdown(
                    label="Sample_Steps", 
                    value=saved_config.get("sample_steps", '32'), 
                    choices=['16','32','48','64','96','128'], 
                    interactive=True, 
                    show_label=True, 
                    allow_custom_value=False
                )
                with gr.Row():
                    self.parallel_infer = gr.Checkbox(
                        label="Parallel_Infer", 
                        value=saved_config.get("parallel_infer", True), 
                        interactive=True, 
                        show_label=True
                    )
                    self.split_bucket = gr.Checkbox(
                        label="Split_Bucket", 
                        value=saved_config.get("split_bucket", True), 
                        interactive=True, 
                        show_label=True
                    )                
                self.how_to_cut = gr.Radio(
                    label=i18n('How to cut'), 
                    choices=list(cut_method.items()), 
                    value=saved_config.get("how_to_cut", list(cut_method.values())[0]), 
                    interactive=True
                )

            with gr.Row():
                self.gen_btn = gr.Button(
                    value=i18n('Generate Audio'), 
                    variant="primary", 
                    visible=True
                )
            
            self.switch_model_btn.click(
                self.switch_model,
                inputs=[self.sovits_path, self.gpt_path, self.api_url],
                outputs=[]
            )

        GPT_SOVITS_ARGS = [
            self.language,
            self.api_url,
            self.refer_audio,
            self.aux_ref_audio,
            self.refer_text,
            self.refer_lang,
            self.batch_size,
            self.batch_threshold,
            self.fragment_interval,
            self.speed_factor,
            self.top_k,
            self.top_p,
            self.temperature,
            self.repetition_penalty,
            self.sample_steps,
            self.parallel_infer,
            self.split_bucket,
            self.how_to_cut,
            self.gpt_path,
            self.sovits_path,
        ]

        # 添加参数记忆功能 - 当任何参数改变时保存配置
        config_inputs = [
            self.language,
            self.api_url,
            self.refer_text,
            self.refer_lang,
            self.batch_size,
            self.batch_threshold,
            self.fragment_interval,
            self.speed_factor,
            self.top_k,
            self.top_p,
            self.temperature,
            self.repetition_penalty,
            self.sample_steps,
            self.parallel_infer,
            self.split_bucket,
            self.how_to_cut
        ]

        # 为每个配置项绑定保存事件
        for component in config_inputs:
            component.change(
                fn=lambda *args: self._save_current_config(*args),
                inputs=config_inputs,
                outputs=[]
            )

        return GPT_SOVITS_ARGS

    def _save_current_config(self, *args):
        """保存当前配置"""
        try:
            config = {
                "language": args[0],
                "api_url": args[1],
                "refer_text": args[2],
                "refer_lang": args[3],
                "batch_size": args[4],
                "batch_threshold": args[5],
                "fragment_interval": args[6],
                "speed_factor": args[7],
                "top_k": args[8],
                "top_p": args[9],
                "temperature": args[10],
                "repetition_penalty": args[11],
                "sample_steps": args[12],
                "parallel_infer": args[13],
                "split_bucket": args[14],
                "how_to_cut": args[15]
            }
            self._save_config(config)
        except Exception as e:
            logger.error(f"保存GPT-SoVITS配置时出错: {e}")

    def arg_filter(self, *args):
        in_file, fps, offset, max_workers, language, api_url, refer_audio, aux_ref_audio, refer_text, refer_lang, batch_size, batch_threshold, fragment_interval, speed_factor, top_k, top_p, temperature, repetition_penalty, sample_steps, parallel_infer, split_bucket, text_split_method, gpt_path, sovits_path = args

        if refer_audio is None:
            gr.Warning(i18n('You must upload Main Reference Audio'))
            raise Exception(i18n('You must upload Main Reference Audio'))

        if refer_audio is not None:
            refer_audio_path = temp_ra(refer_audio)
        else:
            refer_audio_path = ''
        aux_ref_audio_path = [temp_aux_ra(i) for i in aux_ref_audio] if aux_ref_audio is not None else []

        pargs = (
            "GPT_SoVITS",
            dict_language.get(language, language),
            api_url,
            refer_audio_path,
            aux_ref_audio_path,
            refer_text,
            dict_language.get(refer_lang, refer_lang),
            batch_size,
            batch_threshold,
            fragment_interval,
            speed_factor,
            top_k,
            top_p,
            temperature,
            repetition_penalty,
            int(sample_steps),
            parallel_infer,
            split_bucket,
            cut_method.get(text_split_method, text_split_method),
            gpt_path,
            sovits_path
        )
        kwargs = {'in_files': in_file, 'fps': fps, 'offset': offset, 'proj': "gpt_sovits", 'max_workers': max_workers}
        return pargs, kwargs

    def before_gen_action(self, *args, **kwargs):
        force = kwargs.get("force", True)
        notify = kwargs.get("notify", False)
        self.switch_model(gpt_path=args[-2], sovits_path=args[-1], api_url=args[2], force=force, notify=notify)

    def switch_model(self, sovits_path, gpt_path, api_url, force=True, notify=True):
        if self.server_mode:
            if force and notify:
                gr.Warning(i18n('This function has been disabled!'))
            return True

        if not force and sovits_path == self.current_sovits_model.get(api_url) and gpt_path == self.current_gpt_model.get(api_url):
            if notify:
                gr.Info(i18n('Models are not switched. If you need to switch, please manually click the button.'))
            return True

        if sovits_path == "" or gpt_path == "":
            if force and notify:
                gr.Info(i18n('Please specify the model path!'))
            return False

        gr.Info(i18n('Switching Models...'))
        try:
            data_json = {
                "gpt_model_path": gpt_path,
                "sovits_model_path": sovits_path
            }
            response = requests.post(url=f"{api_url}/set_model", json=data_json)
            response.raise_for_status()

            self.current_sovits_model[api_url] = sovits_path
            self.current_gpt_model[api_url] = gpt_path

            if notify:
                gr.Info(i18n('Models switched successfully!'))
            return True
        except Exception as e:
            if notify:
                gr.Warning(f"{i18n('Failed to switch model')}: {e}")
            return False

    def find_models(self):
        """查找可用的模型文件"""
        try:
            # 这里可以根据实际情况实现模型文件扫描
            # 暂时返回空列表
            return gr.update(choices=[]), gr.update(choices=[])
        except Exception as e:
            logger.error(f"查找模型失败: {e}")
            return gr.update(choices=[]), gr.update(choices=[])

    def save_action(self, *args, text: str = None):
        """保存操作，调用API并返回音频数据"""
        try:
            language, api_url, refer_audio, aux_ref_audio, refer_text, refer_lang, batch_size, batch_threshold, fragment_interval, speed_factor, top_k, top_p, temperature, repetition_penalty, sample_steps, parallel_infer, split_bucket, text_split_method, gpt_path, sovits_path = args

            # 准备参考音频路径
            if refer_audio is not None:
                ref_audio_path = temp_ra(refer_audio)
            else:
                ref_audio_path = ''

            # 准备辅助参考音频路径
            aux_ref_audio_paths = [temp_aux_ra(i) for i in aux_ref_audio] if aux_ref_audio is not None else []

            # 调用API
            audio = self.api(
                text_lang=dict_language.get(language, language),
                api_url=api_url,
                ref_audio_path=ref_audio_path,
                aux_ref_audio_paths=aux_ref_audio_paths,
                prompt_text=refer_text,
                prompt_lang=dict_language.get(refer_lang, refer_lang),
                text=text,
                text_split_method=cut_method.get(text_split_method, text_split_method),
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                speed_factor=speed_factor,
                sample_steps=int(sample_steps)
            )
            return audio
        except Exception as e:
            logger.error(f"GPT-SoVITS API调用失败: {e}")
            return None
