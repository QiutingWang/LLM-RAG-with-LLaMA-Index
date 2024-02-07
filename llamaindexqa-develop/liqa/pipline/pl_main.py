import os
import sys
from llama_index import (
    VectorStoreIndex, 
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext, 
    load_index_from_storage
)

from llama_index.node_parser import SimpleNodeParser
pdf_path = "/home/sunshangbin.p/workspace/llamaindexqa/liqa/dataset/right/source"

documents = SimpleDirectoryReader(pdf_path).load_data()

print(len(documents))