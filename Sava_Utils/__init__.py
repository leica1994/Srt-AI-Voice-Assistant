import logging
from collections import defaultdict
import os
import json
import argparse

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-p", "--server_port", type=int, help="server_port")
parser.add_argument("-lan", "--lan", dest="LAN_access", action="store_true", default=False, help="Enable LAN access")
parser.add_argument("--no_ext", dest="no_ext", action="store_true", default=False, help="Do not load any extensions")
parser.add_argument("--share", dest="share", action="store_true", default=False, help="set share True")
parser.add_argument("--server_mode", dest="server_mode", action="store_true", default=False, help="activate server mode")
args, unknown = parser.parse_known_args()
no_ext_mode = args.no_ext

current_path = os.environ.get("current_path")

log_colors = {
    "DEBUG": "white",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}
logger = logging.getLogger("SAVA")
logger.setLevel(logging.INFO)
try:
    import colorlog

    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            fmt="%(log_color)s[%(levelname)s][%(asctime)s]:%(funcName)s: %(message)s",
            datefmt="%Y-%m-%d_%H:%M:%S",
            log_colors=log_colors,
        )
    )
    logger.addHandler(handler)
except ImportError:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s][%(asctime)s]:%(funcName)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from .i18nAuto import I18n

ext_tab_path = os.path.join(current_path, "Sava_Extensions/extensions_config.json")
if os.path.exists(ext_tab_path):
    ext_tab = defaultdict(dict, json.load(open(ext_tab_path, encoding="utf-8")))
else:
    ext_tab = defaultdict(dict)

config_path = os.path.join(current_path, "SAVAdata", "config.json")
try:
    if os.path.isfile(config_path):
        x = json.load(open(config_path, encoding="utf-8"))
        i18n = I18n(x.get("language"))
    else:
        x = dict()
        i18n = I18n()
    from .settings import Settings

    config = Settings.from_dict(x)
    del x
except Exception as e:
    i18n = I18n()
    logger.warning(f"{i18n('Failed to load settings, reset to default')}: {e}")
    from .settings import Settings

    config = Settings()
from .man import Man

MANUAL = Man(language=config.language)

config.server_mode = args.server_mode or config.server_mode
if config.server_mode:
    logger.warning(i18n("Server Mode has been enabled!"))

from .utils import clear_cache

if config.clear_tmp:
    clear_cache()
