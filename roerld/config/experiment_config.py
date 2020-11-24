import os
import json


class ExperimentConfigError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

    @staticmethod
    def missing_key_in_section(key, section):
        return ExperimentConfigError(f"Experiment configuration is missing the required key "
                                     f"'{key}' in section {section}")

    @staticmethod
    def missing_keys(*args):
        return ExperimentConfigError(f"Experiment configuration is missing the required key(s): {','.join(args)}")

    @staticmethod
    def missing_sections(*args):
        return ExperimentConfigError(f"Experiment configuration is missing the required section(s): {','.join(args)}")


class ExperimentConfigView:
    def __init__(self, config: dict, path_to_here):
        self._config = config
        self.path_to_here = path_to_here

    def __getitem__(self, item):
        return self._config.__getitem__(item)

    def section(self, name: str):
        if not self.has_key(name):
            raise ExperimentConfigError.missing_sections(name)

        inner_path = name if not self.path_to_here else self.path_to_here + "." + name

        return ExperimentConfigView(self.key(name), inner_path + "key")

    def sections(self, key: str):
        key_dict = self.key(key)
        return list(key_dict.keys())

    def has_key(self, key_path: str) -> bool:
        try:
            self.key(key_path)
        except ExperimentConfigError:
            return False
        return True

    def key(self, key_path: str):
        """ Retrieves the key.

        :param key_path The key to retrieve. Keys from subsections can be accessed by using
                            'subsection1.subsection2.key'.
        """
        path = str.split(key_path, '.')
        section = self._config
        for i in range(len(path)):
            key = path[i]
            if key not in section:
                path_to_here = ".".join([*self.path_to_here, *path[:i]])
                raise ExperimentConfigError.missing_key_in_section(key, path_to_here)
            section = section[key]

        return section

    def optional_key(self, key_path: str, default):
        """ Retrieves the key.

        :param key_path The key to retrieve. Keys from subsections can be accessed by using
                            'subsection1.subsection2.key'.
        """
        try:
            return self.key(key_path)
        except ExperimentConfigError:
            return default

    def save_to_file(self, path):
        """ Saves the current view as a JSON file.

        :param path The path to save the file to. It may not already exist."""
        assert not os.path.exists(path)
        with open(path, "w") as file:
            config_text = json.dumps(self._config, indent=4)
            file.write(config_text)

    @property
    def path(self):
        return self.path_to_here

    @property
    def config(self):
        return self._config


class ExperimentConfig:
    @staticmethod
    def view(config) -> ExperimentConfigView:
        if type(config) == ExperimentConfigView:
            return config

        return ExperimentConfigView(config, "")
