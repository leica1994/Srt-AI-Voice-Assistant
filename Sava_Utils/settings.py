import os
import gradio as gr
import json
import Sava_Utils
import time
import os
import sys
import platform
import shutil

from . import logger, i18n

current_path = os.environ.get("current_path")

def get_default_gpu_memory():
    """获取GPU显存的默认值 - 针对16GB显卡优化"""
    try:
        import torch
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB

            # 针对不同显存大小的优化策略
            if total_memory >= 15:  # 16GB显卡
                # 16GB显卡使用更保守的70%，确保稳定性
                default_memory = total_memory * 0.7
                logger.info(f"检测到16GB显卡: {total_memory:.1f}GB, 使用保守70%限制: {default_memory:.1f}GB")
            elif total_memory >= 11:  # 12GB显卡
                default_memory = total_memory * 0.75  # 使用75%显存
                logger.info(f"检测到12GB显卡: {total_memory:.1f}GB, 设置默认限制: {default_memory:.1f}GB")
            elif total_memory >= 7:  # 8GB显卡
                default_memory = total_memory * 0.8  # 使用80%显存
                logger.info(f"检测到8GB显卡: {total_memory:.1f}GB, 设置默认限制: {default_memory:.1f}GB")
            else:  # 小显存显卡
                default_memory = total_memory * 0.85  # 使用85%显存
                logger.info(f"检测到小显存显卡: {total_memory:.1f}GB, 设置默认限制: {default_memory:.1f}GB")

            return round(default_memory, 1)
        else:
            logger.info("未检测到CUDA设备，使用CPU模式默认值")
            return 6.0  # CPU模式默认值
    except Exception as e:
        logger.warning(f"GPU显存检测失败: {e}, 使用保守默认值")
        return 6.0  # 出错时的保守默认值

# https://huggingface.co/datasets/freddyaboulton/gradio-theme-subdomains/resolve/main/subdomains.json
gradio_hf_hub_themes = [
    "default",
    "base",
    "glass",
    "soft",
    "gradio/monochrome",
    "gradio/seafoam",
    "gradio/dracula_test",
    "abidlabs/dracula_test",
    "abidlabs/Lime",
    "abidlabs/pakistan",
    "Ama434/neutral-barlow",
    "dawood/microsoft_windows",
    "finlaymacklon/smooth_slate",
    "Franklisi/darkmode",
    "freddyaboulton/dracula_revamped",
    "freddyaboulton/test-blue",
    "gstaff/xkcd",
    "Insuz/Mocha",
    "Insuz/SimpleIndigo",
    "JohnSmith9982/small_and_pretty",
    "nota-ai/theme",
    "nuttea/Softblue",
    "ParityError/Anime",
    "reilnuud/polite",
    "remilia/Ghostly",
    "rottenlittlecreature/Moon_Goblin",
    "step-3-profit/Midnight-Deep",
    "Taithrah/Minimal",
    "ysharma/huggingface",
    "ysharma/steampunk",
    "NoCrypt/miku",
]


class Settings:
    def __init__(
        self,
        language: str = "Auto",
        server_port: int = 8888,
        LAN_access: bool = False,
        overwrite_workspace: bool = False,
        clear_tmp: bool = False,
        concurrency_count: int = 2,
        server_mode: bool = False,
        min_interval: float = 0.3,
        max_accelerate_ratio: float = 1.0,
        output_sr: int = 0,
        remove_silence: bool = False,
        num_edit_rows: int = 7,
        export_spk_pattern: str = "",
        theme: str = "default",
        indextts_pydir: str = "",
        indextts_dir: str = "",
        indextts_args: str = "",
        indextts_script: str = "",
        gsv_fallback: bool = False,
        gsv_pydir: str = "",
        gsv_dir: str = "",
        gsv_args: str = "",
        ms_region: str = "eastasia",
        ms_key: str = "",
        ms_lang_option: str = "zh",
        ollama_url: str = "http://localhost:11434",
        auto_quality_adjustment: bool = True,
        max_gpu_memory_gb: float = None,
    ):
        self.language = language
        self.server_port = int(server_port)
        self.LAN_access = LAN_access
        self.overwrite_workspace = overwrite_workspace
        self.clear_tmp = clear_tmp
        self.concurrency_count = int(concurrency_count)
        self.server_mode = server_mode
        self.min_interval = min_interval
        self.max_accelerate_ratio = max_accelerate_ratio
        self.output_sr = int(output_sr)
        self.remove_silence = remove_silence
        self.num_edit_rows = max(int(num_edit_rows),1)
        self.export_spk_pattern = export_spk_pattern
        self.theme = theme
        self.indextts_pydir = indextts_pydir.strip('"')
        self.indextts_dir = os.path.abspath(indextts_dir.strip('"')) if indextts_dir else indextts_dir
        self.indextts_args = indextts_args
        # 启动脚本可能是命令而不是文件路径，所以不进行路径处理
        self.indextts_script = indextts_script.strip('"') if indextts_script else indextts_script
        self.gsv_fallback = gsv_fallback
        self.gsv_pydir = gsv_pydir.strip('"')
        self.gsv_dir = os.path.abspath(gsv_dir.strip('"')) if gsv_dir else gsv_dir
        self.gsv_args = gsv_args
        self.ms_region = ms_region
        self.ms_key = ms_key
        self.ms_lang_option = ms_lang_option
        self.ollama_url = ollama_url
        self.auto_quality_adjustment = auto_quality_adjustment
        # 如果没有指定显存限制，则自动检测并设置为3/4显存
        self.max_gpu_memory_gb = max_gpu_memory_gb if max_gpu_memory_gb is not None else get_default_gpu_memory()
        # detect python envs####
        if self.indextts_pydir != "":
            if os.path.isfile(self.indextts_pydir):
                self.indextts_pydir = os.path.abspath(self.indextts_pydir)
            elif self.indextts_pydir == 'python':
                pass
            else:
                gr.Warning(f"{i18n('Error, Invalid Path')}:{self.indextts_pydir}")
                self.indextts_pydir = ""
        else:
            if os.path.isfile(os.path.join(current_path, "venv\\python.exe")) and "INDEX" in current_path.upper():
                self.indextts_pydir = os.path.join(current_path, "venv\\python.exe")
                logger.info(f"{i18n('Env detected')}: Index-TTS")
            else:
                self.indextts_pydir = ""

        if self.gsv_pydir != "":
            if os.path.isfile(self.gsv_pydir):
                self.gsv_pydir = os.path.abspath(self.gsv_pydir)
            elif self.gsv_pydir == 'python':
                pass
            else:
                gr.Warning(f"{i18n('Error, Invalid Path')}:{self.gsv_pydir}")
                self.gsv_pydir = ""
        else:
            if os.path.isfile(os.path.join(current_path, "runtime\\python.exe")) and "GPT" in current_path.upper():
                self.gsv_pydir = os.path.join(current_path, "runtime\\python.exe")
                logger.info(f"{i18n('Env detected')}: GPT-SoVITS")
            else:
                self.gsv_pydir = ""
        ###################
        if self.indextts_pydir != "" and indextts_dir == "":
            self.indextts_dir = os.path.dirname(os.path.dirname(self.indextts_pydir))
        if self.gsv_pydir != "" and gsv_dir == "":
            self.gsv_dir = os.path.dirname(os.path.dirname(self.gsv_pydir))

    def to_list(self):
        val = self.to_dict()
        return [val[x] for x in val.keys()]

    def to_dict(self):
        return self.__dict__

    def save(self):
        dic = self.to_dict()
        os.makedirs(os.path.join(current_path, "SAVAdata"), exist_ok=True)
        with open(os.path.join(current_path, "SAVAdata", "config.json"), "w", encoding="utf-8") as f:
            json.dump(dic, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)


def load_cfg():
    config_path = os.path.join(current_path, "SAVAdata", "config.json")
    if os.path.exists(config_path):
        try:
            config = Settings.from_dict(json.load(open(config_path, encoding="utf-8")))
        except Exception as e:
            config = Settings()
            logger.warning(f"Failed to load settings, reset to default: {e}")
    else:
        config = Settings()
    return config


def rm_workspace(name):
    try:
        shutil.rmtree(os.path.join(current_path, "SAVAdata", "workspaces", name))
        gr.Info(f"{name} {i18n('was removed successfully.')}")
        time.sleep(0.1)
        return gr.update(visible=False), gr.update(visible=False)
    except Exception as e:
        gr.Warning(f"{i18n('An error occurred')}: {str(e)}")
        return gr.update(),gr.update()


def restart():
    gr.Warning(i18n('Restarting...'))
    time.sleep(0.5)
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")
    if os.environ.get('exe') != 'True':
        os.execl(sys.executable, f'"{sys.executable}"', f'"{sys.argv[0]}"')
    else:
        try:
            a = os.environ["_PYI_APPLICATION_HOME_DIR"]
            b = os.environ["_PYI_ARCHIVE_FILE"]
            c = os.environ["_PYI_PARENT_PROCESS_LEVEL"]
            os.unsetenv("_PYI_APPLICATION_HOME_DIR")
            os.unsetenv("_PYI_ARCHIVE_FILE")
            os.unsetenv("_PYI_PARENT_PROCESS_LEVEL")
            Sava_Utils.utils.rc_open_window(command=f"{sys.executable}", dir=current_path)
            os.environ["_PYI_APPLICATION_HOME_DIR"] = a
            os.environ["_PYI_ARCHIVE_FILE"] = b
            os.environ["_PYI_PARENT_PROCESS_LEVEL"] = c
        except Exception as e:
            gr.Warning(f"{i18n('An error occurred. Please restart manually!')} {str(e)}")
        os.system(f"taskkill /PID {os.getpid()} /F && exit")


class Settings_UI:
    def __init__(self, componments: list):
        self.componments = componments
        self.ui = False
        self._apply_to_componments()

    def _apply_to_componments(self):
        for item in self.componments.values():
            for i in item:
                i.update_cfg(config=Sava_Utils.config)

    def _get_gpu_info(self):
        """获取GPU信息用于UI显示 - 针对16GB显卡优化"""
        try:
            import torch
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_name = torch.cuda.get_device_name(0)

                # 针对不同显存大小的推荐设置
                if total_memory >= 15:  # 16GB显卡
                    recommended = total_memory * 0.7
                    optimization = "16GB显卡优化: 70%"
                elif total_memory >= 11:  # 12GB显卡
                    recommended = total_memory * 0.75
                    optimization = "12GB显卡优化: 75%"
                elif total_memory >= 7:  # 8GB显卡
                    recommended = total_memory * 0.8
                    optimization = "8GB显卡优化: 80%"
                else:  # 小显存显卡
                    recommended = total_memory * 0.85
                    optimization = "小显存优化: 85%"

                return f"检测到: {gpu_name} ({total_memory:.1f}GB), {optimization}, 推荐: {recommended:.1f}GB"
            else:
                return "未检测到CUDA设备"
        except Exception:
            return "GPU信息获取失败"

    def save_settngs(self, *args):
        current_edit_rows = Sava_Utils.config.num_edit_rows
        Sava_Utils.config = Settings(*args)
        Sava_Utils.config.save()
        self._apply_to_componments()
        if Sava_Utils.config.num_edit_rows != current_edit_rows:
            Sava_Utils.config.num_edit_rows = current_edit_rows
        logger.info(i18n('Settings saved successfully!'))
        gr.Info(i18n('Settings saved successfully!'))
        return Sava_Utils.config.to_list()

    def getUI(self):
        if not self.ui:
            self.ui = True
            return self._UI()
        else:
            raise "ERR"

    def _UI(self):
        if Sava_Utils.config.server_mode:
            gr.Markdown(i18n('Settings have been disabled!'))
            return []
        gr.Markdown(f"⚠️{i18n('Click Apply & Save for these settings to take effect.')}⚠️")
        with gr.Group():
            gr.Markdown(value=i18n('General'))
            self.language = gr.Dropdown(label="Language (Requires a restart)", value=Sava_Utils.config.language, allow_custom_value=False, choices=['Auto', "en_US", "zh_CN", "ja_JP", "ko_KR", "fr_FR"])
            with gr.Row():
                self.server_port = gr.Number(label=i18n('The port used by this program, 0=auto. When conflicts prevent startup, use -p parameter to specify the port.'), value=Sava_Utils.config.server_port, minimum=0)
                self.LAN_access = gr.Checkbox(label=i18n('Enable LAN access. Restart to take effect.'), value=Sava_Utils.config.LAN_access)
            with gr.Row():
                self.overwrite_workspace = gr.Checkbox(label=i18n('Overwrite history records with files of the same name instead of creating a new project.'), value=Sava_Utils.config.overwrite_workspace, interactive=True)
                self.clear_cache = gr.Checkbox(label=i18n('Clear temporary files on each startup'), value=Sava_Utils.config.clear_tmp, interactive=True)
            with gr.Row():
                self.concurrency_count = gr.Number(label=i18n('Concurrency Count'), value=Sava_Utils.config.concurrency_count, minimum=2, interactive=True)
                self.server_mode = gr.Checkbox(label=i18n('Server Mode can only be enabled by modifying configuration file or startup parameters.'), value=Sava_Utils.config.server_mode, interactive=False)
            with gr.Column():
                with gr.Row():
                    self.min_interval = gr.Slider(label=i18n('Minimum voice interval (seconds)'), minimum=0, maximum=3, value=Sava_Utils.config.min_interval, step=0.1)
                    self.max_accelerate_ratio = gr.Slider(label=i18n('Maximum audio acceleration ratio (requires ffmpeg)'), minimum=1, maximum=2, value=Sava_Utils.config.max_accelerate_ratio, step=0.01)
                with gr.Row():
                    self.output_sr = gr.Dropdown(label=i18n('Sampling rate of output audio, 0=Auto'), value='0', allow_custom_value=True, choices=['0', '16000', '22050', '24000', '32000', '44100', '48000'])
                    self.remove_silence = gr.Checkbox(label=i18n('Remove inhalation and silence at the beginning and the end of the audio'), value=Sava_Utils.config.remove_silence, interactive=True)
                with gr.Row():
                    self.num_edit_rows = gr.Number(label=i18n('Edit Panel Row Count (Requires a restart)'), minimum=1, maximum=50, value=Sava_Utils.config.num_edit_rows)
                    self.export_spk_pattern = gr.Text(label=i18n('Export subtitles with speaker name. Fill in your template to enable.'), placeholder=r"{#NAME}: {#TEXT}", value=Sava_Utils.config.export_spk_pattern)

            # 音频处理优化设置
            with gr.Group():
                gr.Markdown(value="🎵 音频处理优化")
                with gr.Row():
                    self.auto_quality_adjustment = gr.Checkbox(
                        label="智能质量调整",
                        value=Sava_Utils.config.auto_quality_adjustment,
                        info="自动根据文件大小调整音频质量，优化大文件处理速度"
                    )
                    # 获取GPU信息用于显示
                    gpu_info = self._get_gpu_info()
                    self.max_gpu_memory_gb = gr.Number(
                        label="GPU显存限制(GB)",
                        value=Sava_Utils.config.max_gpu_memory_gb,
                        minimum=2.0,
                        maximum=24.0,
                        step=0.5,
                        info=f"限制GPU显存使用，避免大文件处理时内存溢出。{gpu_info}"
                    )

            self.theme = gr.Dropdown(choices=gradio_hf_hub_themes, value=Sava_Utils.config.theme, label=i18n('Theme (Requires a restart)'), interactive=True, allow_custom_value=True)
        
        with gr.Accordion(i18n('Storage Management'),open=False):
            self.clear_cache_btn = gr.Button(value=i18n('Clear temporary files'), variant="primary")
            self.clear_cache_btn.click(Sava_Utils.utils.clear_cache, inputs=[], outputs=[])
            self.workspaces_archieves_state = gr.State(value=list())
            self.list_workspaces_btn = gr.Button(value=i18n('List Archives'), variant="primary")
            self.list_workspaces_btn.click(Sava_Utils.edit_panel.refworklist, outputs=[self.workspaces_archieves_state])
            workspaces_manager_ui_empty_md = f"### <center>{i18n('No Archives Found. Click the <List Archives> button to refresh.')}</center>"

            @gr.render(inputs=self.workspaces_archieves_state)
            def workspaces_manager_ui(x: list):
                if len(x) == 0:
                    gr.Markdown(value=workspaces_manager_ui_empty_md)
                    return
                with gr.Group():
                    for i in x:
                        with gr.Row(equal_height=True):
                            item = gr.Textbox(value=i, show_label=False, interactive=False, scale=8)
                            b = gr.Button(value="🗑️", variant="stop", scale=1, min_width=40)
                            b.click(rm_workspace, inputs=[item], outputs=[item, b])

        with gr.Accordion(i18n('Submodule Settings'),open=False):
            with gr.Group():
                gr.Markdown(value="Index-TTS")
                self.indextts_pydir_input = gr.Textbox(label=i18n('Python Interpreter Path for Index-TTS'), interactive=True, value=Sava_Utils.config.indextts_pydir)
                self.indextts_dir_input = gr.Textbox(label=i18n('Root Path of Index-TTS'), interactive=True, value=Sava_Utils.config.indextts_dir)
                self.indextts_args = gr.Textbox(label=i18n('Start Parameters'), interactive=True, value=Sava_Utils.config.indextts_args)
                self.indextts_script = gr.Textbox(label="启动脚本", interactive=True, value=Sava_Utils.config.indextts_script, info="可选：指定启动脚本路径或命令，如果填入则直接执行此脚本/命令")
            with gr.Group():
                gr.Markdown(value="GSV")
                self.gsv_fallback = gr.Checkbox(value=False, label=i18n('Downgrade API version to v1'), interactive=True)
                self.gsv_pydir_input = gr.Textbox(label=i18n('Python Interpreter Path for GSV'), interactive=True, value=Sava_Utils.config.gsv_pydir)
                self.gsv_dir_input = gr.Textbox(label=i18n('Root Path of GSV'), interactive=True, value=Sava_Utils.config.gsv_dir)
                self.gsv_args = gr.Textbox(label=i18n('Start Parameters'), interactive=True, value=Sava_Utils.config.gsv_args)
            with gr.Group():
                gr.Markdown(value=i18n('Translation Module'))
                self.ollama_url = gr.Textbox(label=i18n('Default Request Address for Ollama'), interactive=True, value=Sava_Utils.config.ollama_url)

        self.save_settings_btn = gr.Button(value=i18n('Apply & Save'), variant="primary")
        self.restart_btn = gr.Button(value=i18n('Restart UI'), variant="stop")

        componments_list = [
            self.language,
            self.server_port,
            self.LAN_access,
            self.overwrite_workspace,
            self.clear_cache,
            self.concurrency_count,
            self.server_mode,
            self.min_interval,
            self.max_accelerate_ratio,
            self.output_sr,
            self.remove_silence,
            self.num_edit_rows,
            self.export_spk_pattern,
            self.theme,
            self.indextts_pydir_input,
            self.indextts_dir_input,
            self.indextts_args,
            self.indextts_script,
            self.gsv_fallback,
            self.gsv_pydir_input,
            self.gsv_dir_input,
            self.gsv_args,
            self.ollama_url,
            self.auto_quality_adjustment,
            self.max_gpu_memory_gb,
        ]

        self.save_settings_btn.click(self.save_settngs, inputs=componments_list, outputs=componments_list)
        self.restart_btn.click(restart, [], [])
