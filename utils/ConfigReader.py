import configparser
from typing import List, Dict, Optional


class ConfigReader:
    @classmethod
    def read_section(cls, file_path: str, section: str) -> List[Dict[str, str]]:
        config = configparser.ConfigParser()
        config.read(file_path)
        section_data = []
        if config.has_section(section):
            items = config.items(section)
            for item in items:
                section_data.append({item[0]: item[1]})
        else:
            raise Exception(f"对象'{section}' 不存在")

        return section_data

    def getValueBySection(self, data: List[Dict[str, str]], key: str) -> Optional[str]:
        for item in data:
            if key in item:
                return item[key]
        return None
