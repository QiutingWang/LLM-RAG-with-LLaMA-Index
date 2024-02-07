import re
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import List, Optional
import uuid
from llama_index.finetuning import (
    generate_qa_embedding_pairs,
    EmbeddingQAFinetuneDataset,
)
def plot_histograms(series, output_path=None):
    # 创建一个新的图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))

    # 创建直方图
    sns.histplot(series, kde=False, binwidth=0.05, ax=ax1)

    # 添加标签
    for p in ax1.patches:
        ax1.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')

    # 创建累积直方图
    sns.histplot(series, cumulative=True, binwidth=0.05, stat='density', ax=ax2, element='step')

    # 添加标签
    for p in ax2.patches:
        ax2.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')

    # 调整子图之间的间距
    plt.subplots_adjust(hspace=0.4)

    # 如果指定了输出路径，则将图形保存到该路径
    if output_path:
        plt.savefig(output_path)

    # 显示图形
    plt.show()
def remove_punctuation(input_string):
    # 只保留中文、英文和数字
    output_string = re.sub(r"[^\w\s]", "", input_string)  # \w 包含了英文和数字
    output_string = re.sub(r"\s", "", output_string)  # 删除空格
    return output_string
def check_empty_metadata(nodes):
    """
    查询一组nodes中metadata为空
    """
    empty_metadata_nodes = [node.id for node in nodes if not node.metadata]
    if empty_metadata_nodes != list():
        print("存在空node",empty_metadata_nodes)
    return empty_metadata_nodes
def longest_common_continuous_substring_and_ratio(s1, s2):
    s1,s2 =remove_punctuation(s1),remove_punctuation(s2)
    m = [[0] * (1 + len(s2)) for _ in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    lcs = s1[x_longest - longest: x_longest]
    ratio_s1 = len(lcs) / len(s1)
    ratio_s2 = len(lcs) / len(s2)
    return lcs, ratio_s1, ratio_s2
def longest_common_continuous_substring_and_ratio_batch(list1, list2):
    results = []
    for s1, s2 in zip(list1, list2):
        lcs, ratio_s1, ratio_s2 = longest_common_continuous_substring_and_ratio(s1, s2)
        results.append((lcs, ratio_s1, ratio_s2))
    return results
def generate_qa_embedding_pairs_v2(
    questions: List[str],
    list_nodes=None
) -> EmbeddingQAFinetuneDataset:
    """Generate examples given a set of nodes and questions."""
    node_dict = {
        node.node_id: node.text
      for nodes in list_nodes for node in nodes  
    }
    queries = {}
    for question in questions:
        question_id = str(uuid.uuid4())
        queries[question_id] = question
    # construct dataset
    relevant_docs={question:corpus for question,corpus in zip(list(queries.keys()),[[node.node_id for node in nodes] for nodes in list_nodes])}
    return EmbeddingQAFinetuneDataset(
        queries=queries, corpus=node_dict, relevant_docs=relevant_docs
    )
    
async def evaluate(
    dataset,
    retriever,
    top_k=2,
    verbose=False,
    workers=10
):
    # 初始化检索器
    from llama_index.evaluation import RetrieverEvaluator
    retriever_evaluator = RetrieverEvaluator.from_metric_names(["mrr", "hit_rate"], retriever=retriever)
    
    eval_results = await retriever_evaluator.aevaluate_dataset(dataset,workers=workers)
    
    return eval_results


def display_results(name, eval_results):
    """Display results from evaluate."""

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    hit_rate = full_df["hit_rate"].mean()
    mrr = full_df["mrr"].mean()

    metric_df = pd.DataFrame(
        {"retrievers": [name], "hit_rate": [hit_rate], "mrr": [mrr]}
    )

    return metric_df

def eval_retriever_context(questions,refers,list_nodes):
    top_k=len(list_nodes[0])
    data = []
    for i, nodes in enumerate(list_nodes):
        
        retrieved_files = [Path(node.node.metadata.get('filename') or node.node.metadata.get('file_name')).stem for node in nodes]
   
        row = [questions[i],refers[i]]+ [node.text for node in nodes] +[str(file) for file in retrieved_files]
        data.append(row)
    header=['question','source_context']+[f'retrieved_text{i+1}' for i in range(top_k)]+[f'retrieved_file{i+1}' for i in range(top_k)]
    df_detailed = pd.DataFrame(data,columns=header)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False) # 设置不折叠数据
    pd.set_option('display.max_colwidth', 1000)
    
    return df_detailed