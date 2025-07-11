import locale


class I18n:
    def __init__(self, language=None):
        if language in ["Auto", None]:
            language = locale.getdefaultlocale()[0]
        self.language = language
        ls = dict()
        try:
            exec(f"from .translations.{language} import i18n_dict", globals(), ls)
            self.language_map = ls["i18n_dict"]
        except:
            self.language_map = dict()

    def get_language(self):
        return self.language

    def update(self, i18n_data: dict[str:str]):
        self.language_map.update(i18n_data)

    def __call__(self, key):
        return self.language_map.get(key, key)

    def __repr__(self):
        return f"Using Language: {self.language}"
