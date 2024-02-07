from typing import Callable, Dict, Generator, List, Optional, Type
import os
from llama_index.node_parser import SimpleNodeParser
from llama_index import (
    ServiceContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

from liqa.load.format_pdf_reader import FormatPdfReader, ParaTitle
from liqa.load.format_node_parser import FormatNodeParser
from liqa.load import load_util
from liqa import utils


def _deal_path(input_dir: Optional[str] = None, input_files: Optional[List] = None):
    if input_dir:
        input_dir = utils.convert_path_to_abspath(input_dir)
        storage_dir = input_dir
        input_dir = os.path.join(input_dir, "source")

    if input_files and len(input_files) > 0:
        input_files = utils.convert_path_to_abspath(input_files)
        storage_dir = os.path.dirname(os.path.dirname(input_files[0]))

    storage_dir = os.path.join(storage_dir, "storage", "vec_index")
    return storage_dir, input_dir, input_files


def create_format_vector_index(
    input_dir: Optional[str] = None,
    input_files: Optional[List] = None,
    service_context: ServiceContext = None,
    use_storage: bool = True,
):
    storage_dir, input_dir, input_files = _deal_path(input_dir, input_files)
    if not os.path.exists(storage_dir) or not use_storage:
        documents = SimpleDirectoryReader(
            input_dir=input_dir,
            input_files=input_files,
            file_extractor={".pdf": FormatPdfReader()},
        ).load_data()

        parser = FormatNodeParser.from_defaults(parent_has_child_content=False)
        nodes = parser.get_nodes_from_documents(documents)
        format_nodes= get_leaf_nodes(nodes)
        vector_index = VectorStoreIndex(
            format_nodes, service_context=service_context, show_progess=True
        )
        vector_index.storage_context.persist(persist_dir=storage_dir)
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        vector_index = load_index_from_storage(
            storage_context, service_context=service_context
        )

    return vector_index


def create_simple_vector_index(
    input_dir: Optional[str] = None,
    input_files: Optional[List] = None,
    service_context: ServiceContext = None,
    use_storage: bool = True,
):
    storage_dir, input_dir, input_files = _deal_path(input_dir, input_files)
    if not os.path.exists(storage_dir) or not use_storage:
        documents = SimpleDirectoryReader(
            input_dir=input_dir,
            input_files=input_files,
        ).load_data()

        parser = SimpleNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents(documents)
        vector_index = VectorStoreIndex(
            nodes, service_context=service_context, show_progess=True
        )

        vector_index.storage_context.persist(persist_dir=storage_dir)
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        vector_index = load_index_from_storage(
            storage_context, service_context=service_context
        )

    return vector_index


from liqa.text_spliter import create_text_split
from liqa.load.li_reader import load_docu
from llama_index.node_parser import HierarchicalNodeParser
from llama_index.node_parser import get_leaf_nodes, get_root_nodes
        
pdf_path1 = [
    "liqa/dataset/right/source/中华人民共和国证券法.pdf",
    "liqa/dataset/right/source/保荐人尽职调查工作准则（2022年修订）.pdf",
    "liqa/dataset/right/source/深圳证券交易所证券投资基金交易和申购赎回 实施细则(2019 年修订征求意见稿).pdf",
    "liqa/dataset/right/source/证券公司客户资金账户管理规则.pdf",
    "liqa/dataset/right/source/证券公司融资融券业务管理办法.pdf",
    "liqa/dataset/right/source/证券发行与承销管理办法（2023年修订）.pdf",
    "liqa/dataset/right/source/证券基金经营机构董事、监事、高级管理人员及从业人员监督管理办法.pdf",
    "liqa/dataset/right/source/证券经纪业务管理办法.pdf",
    "liqa/dataset/right/source/首次公开发行股票注册管理办法.pdf",
]

pdf_path2 = [
    "liqa/dataset/right/source/证券期货投资者适当性管理办法.pdf",
    "liqa/dataset/right/source/证券经纪人管理暂行规定（2020年修订）.pdf",
    "liqa/dataset/right/source/境内外证券交易所互联互通存托凭证业务监管规定（2023年修订）.pdf",
]

pattern1 = [
    r"(?:^|\n)第(?:[一二三四五六七八九十零百千万亿]+|[0-9]+)章.*",
    r"(?:^|\n)第(?:[一二三四五六七八九十零百千万亿]+|[0-9]+)条.*",
]

pattern2 = [r"(?:^|\n)第(?:[一二三四五六七八九十零百千万亿]+|[0-9]+)条.*"]

def create_leaf_vector_index(
    input_dir: Optional[str] = None,
    input_files: Optional[List] = None,
    service_context: ServiceContext = None,
    use_storage: bool = True,
):
    storage_dir, input_dir, path1 = _deal_path(input_files=pdf_path1)
    storage_dir, input_dir, path2 = _deal_path(input_files=pdf_path2)
    
    if not os.path.exists(storage_dir) or not use_storage:
        docu1, docu2 = load_docu(path1), load_docu(path2)
        text_splitter_ids1, text_splitter_map1 = create_text_split(patterns=pattern1)
        text_splitter_ids2, text_splitter_map2 = create_text_split(patterns=pattern2)

        parser1 = HierarchicalNodeParser.from_defaults(
            node_parser_ids=text_splitter_ids1, node_parser_map=text_splitter_map1
        )
        parser2 = HierarchicalNodeParser.from_defaults(
            node_parser_ids=text_splitter_ids2, node_parser_map=text_splitter_map2
        )

        nodes = parser1.get_nodes_from_documents(docu1) + parser2.get_nodes_from_documents(docu2)
        leaf_nodes = [node for node in get_leaf_nodes(nodes)]
        vector_index = VectorStoreIndex(leaf_nodes, service_context=service_context, show_progess=True)
        vector_index.storage_context.persist(persist_dir=storage_dir)
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        vector_index = load_index_from_storage(
            storage_context, service_context=service_context
        )

    return vector_index