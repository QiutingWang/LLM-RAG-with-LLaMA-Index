import pandas as pd
from liqa.evaluation.node_checker import Checker
from pathlib import Path
class FileRetrievalStats:
    @staticmethod
    def file_error_rate(df):
        grouped_df = df.groupby('source_file')['search'].agg(['mean', 'count']).reset_index().sort_values('count', ascending=False)
        grouped_df['mean'] *= 100
        grouped_df['mean'] = grouped_df['mean'].round(2)
        return grouped_df
    @staticmethod
    def confuse_pair(df):
        filtered_df = df[df['search'] == False]
        grouped_df = filtered_df.groupby(['retrieved_file1', 'source_file']).size().reset_index(name='count')
        grouped_df = grouped_df.sort_values('count', ascending=False).reset_index(drop=True)
        return grouped_df
    @classmethod
    def analyze(cls, df):
        error_rate = cls.file_error_rate(df)
        confuse_pairs = cls.confuse_pair(df)
        return error_rate, confuse_pairs

def eval_retriever_file(retriever, top_k,file_name='test_large.xlsx'):
    # 读取 Excel 文件
    df = pd.read_excel(file_name)
    questions = list(df['question'])
    refers = list(df['response'])
    source_file=list(df['source_file'])
    

    # 使用 retriever 检索节点
    list_refers = []
    is_infile = []
    data = []
    for i, question in enumerate(questions):
        nodes = retriever.retrieve(question)
        Checker.empty_metadata([node.node for node in nodes])
        list_refers.append(nodes)
        
        retrieved_files = [Path(node.node.metadata.get('filename') or node.node.metadata.get('file_name')).stem for node in nodes]
        is_infile.append(source_file[i] in retrieved_files)
        
        row = [question,source_file[i] in retrieved_files, source_file[i]] +[refers[i]] +[str(file) for file in retrieved_files]+ [node.text for node in nodes]
        data.append(row)

    print('文件检索准确率',round(sum(is_infile)/150, 2)
)

    # 生成并保存DataFrame表格
    header=['Question','search', 'source_file','source_context']+[f'retrieved_file{i+1}' for i in range(top_k)]+[f'retrieved_text{i+1}' for i in range(top_k)]
    df_detailed = pd.DataFrame(data,columns=header)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    return df_detailed

def calculate_percentage(docu, leaf_nodes):
    # 获取文档名称
    doc_names = [doc.metadata.get('file_name') for doc in docu]

    # 计算原始文档的长度
    source_docu = [len(doc.text) for doc in docu]

    # 计算处理后的文档的长度
    post_docu = [sum([len(node.text) for node in Checker.nodes_file(leaf_nodes, filename=name)]) for name in doc_names]

    # 计算百分比
    percentage = [(x / y) * 100 for x, y in zip(post_docu, source_docu)]

    # 创建一个包含文档名称和百分比的列表
    result = list(zip(doc_names, percentage))

    return pd.DataFrame(result, columns=['Document Name', 'Percentage'])

def process_and_plot(form_save_regex, leaf_save_regex, error_rate3, error_rate2):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    error_rate3['source_file'] = error_rate3['source_file'].apply(lambda x: x+'.pdf')
    error_rate2['source_file'] = error_rate2['source_file'].apply(lambda x: x+'.pdf')

    merged_df = form_save_regex.merge(leaf_save_regex, left_on="Document Name", right_on="Document Name")
    merged_df = merged_df.merge(error_rate3, left_on="Document Name", right_on="source_file")
    merged_df = merged_df.merge(error_rate2, left_on="Document Name", right_on="source_file")

    df = merged_df[['Document Name', 'Percentage_x', 'Percentage_y','mean_x', 'mean_y']]

    # 将数据集从宽格式转换为长格式
    df_long = pd.melt(df, id_vars='Document Name', value_vars=['Percentage_x', 'Percentage_y', 'mean_x', 'mean_y'], 
                      var_name='Metric', value_name='Value')

    # 创建一个新的列，将 'Percentage' 和 'mean' 分开
    df_long['Metric Type'] = df_long['Metric'].apply(lambda x: 'Percentage' if 'Percentage' in x else 'mean')

    # 创建分组折线图
    g = sns.FacetGrid(df_long, col='Metric Type', height=5, aspect=1.5)
    g.map(sns.lineplot, 'Document Name', 'Value', 'Metric', marker='o')
    # 添加图例
    g.add_legend()
    plt.xticks([])

    plt.show()

def get_unique_elements(list1, list2):
    # 将两个列表转换为集合
    set1 = set(list1)
    set2 = set(list2)

    # 找到两个集合的差集，即在set1中但不在set2中的元素
    diff1 = set1 - set2

    # 找到两个集合的差集，即在set2中但不在set1中的元素
    diff2 = set2 - set1

    # 合并两个差集，得到两个列表中不重复的部分
    result = diff1.union(diff2)

    return diff1,diff2

