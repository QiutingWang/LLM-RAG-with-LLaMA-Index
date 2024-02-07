import pandas as pd
import nest_asyncio
nest_asyncio.apply()
import asyncio

import logging
import sys
import os

from llama_index.llms import OpenAI
from llama_index.evaluation import SemanticSimilarityEvaluator, FaithfulnessEvaluator
from llama_index import (
    ServiceContext,
)

liqa_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(liqa_dir)

from llama_index.embeddings import SimilarityMode, HuggingFaceEmbedding
from liqa.load.format_pdf_reader import FormatPdfReader, ParaTitle
from liqa.load.format_node_parser import FormatNodeParser
from liqa.load import load_util
from liqa.query.chinese_prompt import DEFAULT_TEXT_QA_PROMPT, DEFAULT_REFINE_PROMPT
from liqa.pipline import pipline_utils
from liqa import utils

utils.default_env()

llm = OpenAI(
    model="gpt-3.5-turbo",
    temperature=0.1,
    api_key="sk-aJzbu0F3j7bstWlR3e4cA9Db59Ac4f669a9f471aFa66C458",
    api_base="https://gf.nekoapi.com/v1",
)
# embed_model = "local:BAAI/bge-large-zh"
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-zh",
    cache_folder="/nvme/share/share/yangyihe/embedding",
    embed_batch_size=3,
)
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

pdf_path1 = ["liqa/dataset/right/source/中华人民共和国证券法.pdf"]
vector_index = pipline_utils.create_vector_index(input_files=pdf_path1, service_context=service_context)


def reference_response_file(query_engine):
    # 得到question和对应的references
    df = pd.read_excel(utils.convert_path_to_abspath("liqa/dataset/test_large.xlsx"))
    questions = list(df["question"])
    references = list(df["response"])

    # 制作表格
    table = pd.DataFrame(
        {
            "Questions": questions[0:5],
            "References": references[0:5]
        }
    )
    return table


def get_responses(table, query_engine):
    respon_list = []
    for _, (question, _) in table.iterrows():
        response = query_engine.query(question)
        respon_list.append(response)
    return respon_list

src_dataframe = reference_response_file()
get_responses(src_dataframe, vector_index.as_query_engine(text_qa_template=DEFAULT_TEXT_QA_PROMPT, refine_template=DEFAULT_REFINE_PROMPT))


results = pd.DataFrame(
    {
        "Responses": get_responses(src_dataframe),
    }
)
display = pd.concat([src_dataframe, results], axis=1)


def similarity_score(dataframe):
    embed_model1 = HuggingFaceEmbedding(
        model_name="BAAI/bge-large-zh",
        cache_folder="/nvme/share/share/yangyihe/embedding",
        embed_batch_size=3,
        text_instruction="",
    )
    service_context1 = ServiceContext.from_defaults(llm=llm, embed_model=embed_model1)

    evaluator1 = SemanticSimilarityEvaluator(service_context=service_context1)
    sc_list = []
    for i in range(5):
        result = asyncio.run(
            evaluator1.aevaluate(
                response=dataframe.Responses[i], reference=dataframe.References[i]
            )
        )
        sc_list.append(result.score)
    return sc_list


def Faithfulness_test(dataframe):
    evaluator2 = FaithfulnessEvaluator(service_context=service_context)
    score_list = []
    for i in range(5):
        result = evaluator2.evaluate_response(response=dataframe.Responses[i])
        score_list.append(result.score)
    return score_list


def display_evaluation_results(dataframe):
    results = pd.DataFrame(
        {
            "Similarity Score": similarity_score(dataframe),
            "Faithfulness Score": Faithfulness_test(dataframe),
        }
    )
    display = pd.concat([dataframe, results], axis=1)
    return display


def main(dataframe):
    similarity_score(dataframe)
    Faithfulness_test(dataframe)
    # await display_evaluation_results(dataframe)
    dataframe2 = display_evaluation_results(dataframe)
    return dataframe2


src_dataframe_path = "src_dataframe.csv"
if os.path.exists(src_dataframe_path):
    src_dataframe = pd.read_csv("src_dataframe.csv")
else:
    src_dataframe = reference_response_file(vector_index.as_query_engine())
    src_dataframe.to_csv("src_dataframe.csv")

main(src_dataframe)
