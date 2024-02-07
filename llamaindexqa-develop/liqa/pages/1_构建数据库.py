import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import os
from dataclasses import dataclass, asdict
import streamlit_antd_components as sac
import faiss
from utils import get_pdf_text,get_text_chunks,ChatAssistant
import pandas as pd
import streamlit_antd_components as sac
from streamlit_antd_components.utils.data_class import BsIcon
from config import config
from llama_index import ServiceContext, set_global_service_context
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding, HuggingFaceEmbedding
from llama_index.node_parser import SentenceWindowNodeParser, SimpleNodeParser
from llama_index import (
    ServiceContext, # 定义了管道式使用的一组服务和配置
VectorStoreIndex
)

from llama_index.embeddings import OpenAIEmbedding
from llama_index import SimpleDirectoryReader
import tempfile

def save_uploaded_files(uploaded_files,dataset_name):
    saved_files = []
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            with open(os.path.join(f"liqa/dataset/{dataset_name}/source/", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append(uploaded_file.name)
    return saved_files
import llama_index

llama_index.set_global_handler("simple")


st.header("构建数据库")

pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
dataset_name=st.text_input("设定数据库名称")

if st.button("载入",use_container_width=True):
    st.info('开始构建', icon="ℹ️")
    
    
    ## 模型配置
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1,api_key='sk-aJzbu0F3j7bstWlR3e4cA9Db59Ac4f669a9f471aFa66C458',api_base='https://gf.nekoapi.com/v1')
    embed_model=OpenAIEmbedding(model="text-embedding-ada-002",api_key='sk-aJzbu0F3j7bstWlR3e4cA9Db59Ac4f669a9f471aFa66C458',api_base='https://gf.nekoapi.com/v1')
    ## 文本分割器
    # 节点解析器

    node_parser = SimpleNodeParser.from_defaults(
        # text_splitter=text_splitter,
        # metadata_extractor=metadata_extractor
    )

    ctx = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    if not os.path.exists(f"liqa/dataset/{dataset_name}"):
        os.makedirs(f"liqa/dataset/{dataset_name}")
    if not os.path.exists(f"liqa/dataset/{dataset_name}/source"):
        os.makedirs(f"liqa/dataset/{dataset_name}/source")
    with st.status("构建数据库"):
        saved_files = save_uploaded_files(pdf_docs,dataset_name )
        if saved_files:
            st.write(f"保存：{saved_files[0]}...")
        documents= SimpleDirectoryReader(
            input_dir='liqa/dataset/'+dataset_name+'/source',
        )
        docs = documents.load_data()
        
        base_nodes = node_parser.get_nodes_from_documents(docs)

        st.write(f"构建索引")
        base_index = VectorStoreIndex(base_nodes, service_context=ctx,use_async=True,show_progess=True)
        base_index.storage_context.persist('liqa/dataset/'+dataset_name+'/storage')
        
    st.toast('数据库构建完成', icon='😍')
        # #保存sorce
    
    
 
