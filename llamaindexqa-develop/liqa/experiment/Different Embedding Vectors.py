#  直接计算余弦相似度
#  conclusion:部分Embedding模型需要Instruction才能启动，而get_text_embedding 和get_query_embedding模型的Instruction不一致

from llama_index.embeddings.huggingface_utils import DEFAULT_HUGGINGFACE_EMBEDDING_MODEL,DEFAULT_QUERY_BGE_INSTRUCTION_ZH 
from llama_index.schema import TextNode
from llama_index import VectorStoreIndex,ServiceContext
import os 
from llama_index.llms import OpenAI
from llama_index.embeddings import HuggingFaceEmbedding
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
os.environ['TRANSFORMERS_CACHE'] = '/nvme/share/share/yangyihe/embedding'
import numpy as np
def get_similarity(embedding1:str, embedding2:str):
    product = np.dot(embedding1, embedding2)
    norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    return product / norm

embed_model =HuggingFaceEmbedding(model_name="BAAI/bge-large-zh", cache_folder='/nvme/share/share/yangyihe/embedding',embed_batch_size=3)

test_str_right="""
证券公司根据投资者的委托，按照证券交易规则提出交易申报，参与证券交易所场内的集中交易，并根据成交结果承担相应的清算交收责任。证券登记结算机构根据成交结果，按照清算交收规则，与证券公司进行证券和资金的清算交收，并为证券公司客户办理证券的登记过户手续。 
"""
test_str_wrong="""
中国证监会依法对证券发行与承销行为进行监督管理。证券交易所、证券登记结算机构和中国证券业协会应当制定相关业务规则，规范证券发行与承销行为。
中国证监会依法批准证券交易所制定的发行承销制度规则，建立对证券交易所发行承销过程监管的监督机制，持续关注证券交易所发行承销过程监管情况。
证券交易所对证券发行承销过程实施监管，对发行人及其控股股东、实际控制人、董事、监事、高级管理人员，承销商、证券服务机构、投资者等进行自律管理。
中国证券业协会负责对承销商、网下投资者进行自律管理。 
"""

embeddings=embed_model.get_text_embedding_batch([test_str_right,test_str_wrong])
embedding1,embedding2=embed_model.get_text_embedding(test_str_right),embed_model.get_text_embedding(test_str_wrong)
query_embedding=embed_model.get_text_embedding("在证券发行承销过程中，交易所承担什么角色？")
query_embedding2=embed_model.get_query_embedding("在证券发行承销过程中，交易所承担什么角色？")
# 将所有的嵌入向量放在一个列表中
all_embeddings = [embeddings[0], embeddings[1], embedding1, embedding2]

# 对每个嵌入向量，计算它与 query_embedding 的余弦相似度
for i, emb in enumerate(all_embeddings):
    similarity = get_similarity(query_embedding, emb)
    print(f"Similarity with embedding {i+1}: {similarity}")

# 对每个嵌入向量，计算它与 query_embedding 的余弦相似度
for i, emb in enumerate(all_embeddings):
    similarity = get_similarity(query_embedding2, emb)
    print(f"Similarity with embedding {i+1}: {similarity}") 