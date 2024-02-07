from typing import Dict, List, Optional

import re
import asyncio
import pandas as pd
from pandas import DataFrame
from pathlib import Path
import json

from llama_index.response.schema import Response, NodeWithScore
from llama_index.evaluation import RelevancyEvaluator, FaithfulnessEvaluator
from llama_index import ServiceContext
from llama_index.embeddings import SimilarityMode, HuggingFaceEmbedding
from llama_index.evaluation import SemanticSimilarityEvaluator, FaithfulnessEvaluator
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.response.schema import RESPONSE_TYPE
from llama_index import QueryBundle
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.retrievers import BaseRetriever

from liqa import utils
from liqa.evaluation.node_checker import Checker

class RerankRetriever(BaseRetriever):
    def __init__(self, vector_retriever, reranker):
        self.vector_retriever = vector_retriever
        self.reranker=reranker

    def _retrieve(self, query, **kwargs):
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)
        all_nodes = []
        node_ids = set()
        for n in vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        all_nodes=self.reranker.postprocess_nodes(all_nodes,query_bundle=query)
        print("RerankRetriever")
        return all_nodes

class EvalTool:
    Key_Questions = "Questions"
    Key_References = "References"
    Key_Responses = "Responses"
    Key_Faithfulness = "Faithfulness"
    Key_Similarity = "Similarity"
    Key_Metadata = "Metadata"
    Key_SF = "SourceFiles"
    Key_InFile = "RetrieverInFile"
    Key_Srouce = "Srouce"
    
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-large-zh",
        # model_name = 'moka-ai/m3e-large',
        cache_folder ="/nvme/share/share/yangyihe/embedding",
        embed_batch_size=3,
    )

    service_context = ServiceContext.from_defaults(embed_model=embed_model)
    evaluator_faithfulness = FaithfulnessEvaluator(service_context=service_context)
    evaluator_similarity = SemanticSimilarityEvaluator(service_context=service_context)
        
    @staticmethod
    def get_reranker(vector_index, top_k=2, similarity_top_k=6):
        reranker = SentenceTransformerRerank(top_n=top_k, model="BAAI/bge-reranker-large")
        retriever = vector_index.as_retriever(similarity_top_k=10)
        hybrid_retriever = RerankRetriever(retriever, reranker)
        return hybrid_retriever
    
    
    @staticmethod
    def get_node_file(node:NodeWithScore):
        return Path(node.node.metadata.get('filename') or node.node.metadata.get('file_name')).stem

    @staticmethod
    def read_source_table():
        df = pd.read_excel(
            utils.convert_path_to_abspath("liqa/dataset/test_large.xlsx")
        )
        questions = list(df["question"])
        references = list(df["response"])
        source_files = list(df["source_file"])
        references = [re.sub("根据.*仅供参考。", "", q).strip() for q in references]

        table = pd.DataFrame(
            {
                EvalTool.Key_Questions: questions,
                EvalTool.Key_References: references,
                EvalTool.Key_SF: source_files,
            }
        )
        return table
    
    @staticmethod
    def construct_source_content(nodes: List[NodeWithScore], with_source_info: bool = True):
        if with_source_info:
            return "\n".join([f"##{node.score}, 【{EvalTool.get_node_file(node)}】, {node.get_content()}" for node in nodes])
        else:
            return "\n".join([f"{node.get_content()}" for node in nodes])
        
    @staticmethod
    async def query_engine_aquery_batch(query_engine: BaseQueryEngine, questions: List[str], batch_size: int = 80) -> RESPONSE_TYPE:
        task_list = []
        for i in range(0, len(questions), batch_size):
            task_list = [query_engine.aquery(question) for question in questions[i: min(i + batch_size, len(questions))]]
            for rst in await asyncio.gather(*task_list):
                yield rst

    @staticmethod
    async def construct_response_table(query_engine: BaseQueryEngine, dataframe:DataFrame, with_source: bool = True, with_source_info: bool = True) -> pd.DataFrame:
        data = []
        curt_index = 0
        
        async for response in EvalTool.query_engine_aquery_batch(query_engine, dataframe.Questions.tolist()):
            try:
                curt_dict = {
                    EvalTool.Key_Responses: json.loads(response.response)['choices'][0]['message']['content']
                }
            except:
                curt_dict = {
                    EvalTool.Key_Responses: response.response
                }
            
            if with_source:
                curt_dict.update(
                    {
                        EvalTool.Key_Srouce : EvalTool.construct_source_content(response.source_nodes, with_source_info)
                    }
                )
            
            data.append(curt_dict)
            curt_index = curt_index + 1
            if curt_index % 30 == 0:
                print(f"construct_response_table query_engine id:{id(query_engine)}, curt_index:{curt_index}")
        
        return pd.concat([dataframe, pd.DataFrame(data)], axis=1)

    @staticmethod
    def construct_retriever_table(retriever, dataframe:DataFrame, with_source: bool = True, with_source_info: bool = True):
        data = []
        for question, source_file in zip(dataframe[EvalTool.Key_Questions], dataframe[EvalTool.Key_SF]):
            nodes = retriever.retrieve(question)
            
            retrieved_files = [EvalTool.get_node_file(node) for node in nodes]
            curt_dict = {
                EvalTool.Key_InFile: source_file in retrieved_files
            }
            
            if with_source:
                curt_dict.update(
                    {
                        EvalTool.Key_Srouce : EvalTool.construct_source_content(nodes, with_source_info)
                    }
                )
            
            data.append(curt_dict)
            
        return pd.concat([dataframe, pd.DataFrame(data)], axis=1)


    @staticmethod
    def similarity_score(reference: str, response: str):
        result = asyncio.run(
            EvalTool.evaluator_similarity.aevaluate(
                response=response, reference=reference
            )
        )
        return result.score

    @staticmethod
    def Faithfulness_test(response: Optional[Response]):
        result = EvalTool.evaluator_faithfulness.evaluate_response(
            response=response
        )
        return result.score
