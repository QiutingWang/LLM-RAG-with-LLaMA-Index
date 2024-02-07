from llama_index.text_splitter import SentenceSplitter,TokenTextSplitter
from liqa.text_spliter import MyTextSpliter
def create_text_split(patterns):
    '''
    自己设定的spliter，最底层加上Token限制
    patterns = [
        r'(?:^|\n)第(?:[一二三四五六七八九十零百千万亿]+|[0-9]+)章.*',
        r'(?:^|\n)第(?:[一二三四五六七八九十零百千万亿]+|[0-9]+)条.*',
    ]
    '''
    text_splitter_ids = []
    text_splitter_map = {}
    for level, pattern in enumerate(patterns):
        split_id = f"level_{level}"
        text_splitter_ids.append(split_id)
        text_splitter_map[split_id] = MyTextSpliter(pattern=pattern, level=level)

    four_splitter = SentenceSplitter()
    text_splitter_ids.append("level_{}".format(len(pattern)))
    text_splitter_map["level_{}".format(len(pattern))] = four_splitter

    return text_splitter_ids,text_splitter_map 