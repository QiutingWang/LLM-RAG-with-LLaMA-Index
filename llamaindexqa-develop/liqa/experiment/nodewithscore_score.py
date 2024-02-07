# 用来debug Score为什么分数较低
from llama_index.schema import TextNode
from llama_index import VectorStoreIndex,ServiceContext
import os 
from llama_index.llms import OpenAI
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.vector_stores import SingleStoreVectorStore,VectorStoreQuery,VectorStoreQueryResult
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
os.environ['TRANSFORMERS_CACHE'] = '/nvme/share/share/yangyihe/embedding'
import numpy as np
def get_similarity(embedding1:str, embedding2:str):
    product = np.dot(embedding1, embedding2)
    norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    return product / norm

embed_model =HuggingFaceEmbedding(model_name="BAAI/bge-large-zh", cache_folder='/nvme/share/share/yangyihe/embedding',embed_batch_size=7)
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1,api_key='sk-aJzbu0F3j7bstWlR3e4cA9Db59Ac4f669a9f471aFa66C458',api_base='https://gf.nekoapi.com/v1')
# embed_model=OpenAIEmbedding
ctx = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model
)
test_str_right="""
证券公司根据投资者的委托，按照证券交易规则提出交易申报，参与证券交易所场内的集中交易，并根据成交结果承担相应的清算交收责任。证券登记结算机构根据成交结果，按照清算交收规则，与证券公司进行证券和资金的清算交收，并为证券公司客户办理证券的登记过户手续。 
"""
test_str_wrong="""
中国证监会依法对证券发行与承销行为进行监督管理。证券交易所、证券登记结算机构和中国证券业协会应当制定相关业务规则，规范证券发行与承销行为。
中国证监会依法批准证券交易所制定的发行承销制度规则，建立对证券交易所发行承销过程监管的监督机制，持续关注证券交易所发行承销过程监管情况。
证券交易所对证券发行承销过程实施监管，对发行人及其控股股东、实际控制人、董事、监事、高级管理人员，承销商、证券服务机构、投资者等进行自律管理。
中国证券业协会负责对承销商、网下投资者进行自律管理。 
"""
nodes=[TextNode(text=test_str_right,node_id='right'),TextNode(text=test_str_wrong,node_id='wrong')]
index=VectorStoreIndex(nodes,service_context=ctx)
vector_store=index._vector_store
from llama_index.vector_stores import VectorStoreQueryResult
query_str = "在证券发行承销过程中，交易所承担什么角色？"

query_embedding = embed_model.get_query_embedding(query_str)
# construct vector store query
query_mode = "default"
# query_mode = "sparse"
# query_mode = "hybrid"

vector_store_query = VectorStoreQuery(
    query_embedding=query_embedding, similarity_top_k=2, mode=query_mode
)
# returns a VectorStoreQueryResult
query_result = vector_store.query(vector_store_query)
query_result