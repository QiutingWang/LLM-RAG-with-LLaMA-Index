import re
from llama_index.node_parser import TextSplitter
from typing import List
class MyTextSpliter(TextSplitter):
    pattern: str = None
    level: int = 0

    @classmethod
    def class_name(cls) -> str:
        return "MyTextSpliter"

    def split_text(self, text: str) ->  List[str]:
        # 使用你的正则表达式规则来分割文本，同时保留分割符（章节标题）
        splits = re.split(self.pattern, text)
        titles = re.findall(self.pattern, text)
        # 将章节标题和相应的内容组合在一起
        result = [titles[i] + splits[i+1] for i in range(len(titles))]
        return result
