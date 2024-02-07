from llama_index.node_parser import SimpleNodeParser,HierarchicalNodeParser
from liqa.load.li_reader import load_docu
from liqa.text_spliter import MyTextSpliter,create_text_split
from llama_index.text_splitter import SentenceSplitter
from liqa.evaluation.node_checker import Checker
from llama_index import ServiceContext
from llama_index.llms import OpenAI
from llama_index.embeddings import OpenAIEmbedding
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.node_parser import get_leaf_nodes, get_root_nodes
from liqa.load.format_pdf_reader import FormatPdfReader, ParaTitle
from liqa.load.format_node_parser import FormatNodeParser
from liqa.load import load_util
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from liqa.evaluation.retriever_file import eval_retriever_file
from liqa.evaluation.retriever_file import FileRetrievalStats
import torch
import os
def main(i_th):
    print("开始第 {} 轮".format(i_th))
    import os

    os.environ["http_proxy"] = "http://youhongming.p:you19980819*@10.1.8.50:33128/"
    os.environ["https_proxy"] = "http://youhongming.p:you19980819*@10.1.8.50:33128/"
    os.environ["HTTP_PROXY"] = "http://youhongming.p:you19980819*@10.1.8.50:33128/"
    os.environ["HTTPS_PROXY"] = "http://youhongming.p:you19980819*@10.1.8.50:33128/"
    # 基于不同的检索规则分别加载文档，分别进行节点解析，解析完的节点再进行合并
    pdf_path1=['liqa/dataset/right/source/中华人民共和国证券法.pdf',
    'liqa/dataset/right/source/保荐人尽职调查工作准则（2022年修订）.pdf',
    'liqa/dataset/right/source/深圳证券交易所证券投资基金交易和申购赎回 实施细则(2019 年修订征求意见稿).pdf',
    'liqa/dataset/right/source/证券公司客户资金账户管理规则.pdf',
    'liqa/dataset/right/source/证券公司融资融券业务管理办法.pdf',
    'liqa/dataset/right/source/证券发行与承销管理办法（2023年修订）.pdf',
    'liqa/dataset/right/source/证券基金经营机构董事、监事、高级管理人员及从业人员监督管理办法.pdf',
    'liqa/dataset/right/source/证券经纪业务管理办法.pdf',
    'liqa/dataset/right/source/首次公开发行股票注册管理办法.pdf']


    pdf_path2=['liqa/dataset/right/source/证券期货投资者适当性管理办法.pdf',
    'liqa/dataset/right/source/证券经纪人管理暂行规定（2020年修订）.pdf',
    'liqa/dataset/right/source/境内外证券交易所互联互通存托凭证业务监管规定（2023年修订）.pdf']

    pattern1= [
            r'(?:^|\n)第(?:[一二三四五六七八九十零百千万亿]+|[0-9]+)章.*',
            r'(?:^|\n)第(?:[一二三四五六七八九十零百千万亿]+|[0-9]+)条.*'
        ]

    pattern2= [
            r'(?:^|\n)第(?:[一二三四五六七八九十零百千万亿]+|[0-9]+)条.*'
        ]


    embedding_list=[
    "BAAI/bge-large-zh",
    "BAAI/bge-large-zh-v1.5",
    "infgrad/stella-base-zh",
    "infgrad/stella-base-zh-v2",
    "infgrad/stella-large-zh-v2",
    "sensenova/piccolo-large-zh",
    "thenlper/gte-base-zh",
    "thenlper/gte-large-zh",
    "infgrad/stella-large-zh",
    "amu/tao",
    "amu/tao-8k",
    ]

    
    
    docu1,docu2=load_docu(pdf_path1),load_docu(pdf_path2)
    text_splitter_ids1,text_splitter_map1 = create_text_split(patterns=pattern1)
    text_splitter_ids2,text_splitter_map2 = create_text_split(patterns=pattern2)

    parser1 = HierarchicalNodeParser.from_defaults(text_splitter_ids=text_splitter_ids1,text_splitter_map=text_splitter_map1)
    parser2 = HierarchicalNodeParser.from_defaults(text_splitter_ids=text_splitter_ids2,text_splitter_map=text_splitter_map2)

    nodes = parser1.get_nodes_from_documents(docu1)+parser2.get_nodes_from_documents(docu2)
    ## 加载环境
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    os.environ['TRANSFORMERS_CACHE'] = '/nvme/share/share/yangyihe/embedding'
    os.environ['HF_HOME'] = '/nvme/share/share/yangyihe/embedding'
    os.environ["LLAMA_INDEX_CACHE_DIR"] = '/nvme/share/share/yangyihe/embedding'
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1,api_key='sk-aJzbu0F3j7bstWlR3e4cA9Db59Ac4f669a9f471aFa66C458',api_base='https://gf.nekoapi.com/v1')
    
    # embed_model="local:{}".format()
    embed_model = HuggingFaceEmbedding(
                model_name=embedding_list[i_th], cache_folder='/nvme/share/share/yangyihe/embedding',embed_batch_size=3
            )

    
    # embed_model=OpenAIEmbedding
    ctx = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model
    )

    # 加载retriever
    
    leaf_nodes=[node for node in get_leaf_nodes(nodes)]
    leaf_index=VectorStoreIndex(leaf_nodes, service_context=ctx,show_progess=True)

    # 对照retriever
    sim_documents= SimpleDirectoryReader(input_dir='liqa/dataset/right/source').load_data()
    sim_node_parser = SimpleNodeParser.from_defaults()
    sim_nodes = sim_node_parser.get_nodes_from_documents(sim_documents, show_progress=False)
    sim_index = VectorStoreIndex(sim_nodes,service_context=ctx,show_progess=True)


    
    documents = SimpleDirectoryReader(input_dir='liqa/dataset/right/source', file_extractor={".pdf": FormatPdfReader()}).load_data()
    parser = FormatNodeParser.from_defaults(parent_has_child_content=False)
    nodes= parser.get_nodes_from_documents(documents)
    format_nodes= [node for node in get_leaf_nodes(nodes)]
    format_index = VectorStoreIndex(format_nodes,service_context=ctx,show_progess=True)


    

    top_k=1

    sim_retriever = sim_index.as_retriever(similarity_top_k=top_k)
    ans1=eval_retriever_file(retriever=sim_retriever,top_k=top_k)
    error_rate1, confuse_pairs1 = FileRetrievalStats.analyze(df=ans1)
    leaf_retriever = leaf_index.as_retriever(similarity_top_k=top_k)
    ans2=eval_retriever_file(retriever=leaf_retriever,top_k=top_k)
    error_rate2, confuse_pairs2 = FileRetrievalStats.analyze(df=ans2)

    format_retriever = format_index.as_retriever(similarity_top_k=top_k)
    ans3=eval_retriever_file(retriever=format_retriever,top_k=top_k)
    error_rate3, confuse_pairs3 = FileRetrievalStats.analyze(df=ans3)
    del ctx
    del embed_model
    torch.cuda.empty_cache()
if __name__=="__main__":
    
    for i in range(11):
        main(i+4)  # 调用main函数