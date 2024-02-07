import streamlit as st
import streamlit_antd_components as sac
import os
import csv
import base64
from streamlit_javascript import st_javascript
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_echarts import st_echarts
import pandas as pd
import numpy  as np
from streamlit_option_menu import option_menu
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import shutil
from llama_index.storage.docstore import SimpleDocumentStore
st.set_page_config(
    page_title="Sensetime Dqa demo App",
    layout="wide",
    initial_sidebar_state="expanded"
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def get_filenames_without_extension(directory):
    filenames_with_extension = os.listdir(directory)
    filenames_without_extension = [os.path.splitext(filename)[0] for filename in filenames_with_extension]
    return filenames_without_extension


def generate_wordcloud_from_csv(csv_file):
    # 读取上传的CSV文件
    df = pd.read_csv(csv_file)

    # 确保CSV文件至少包含一个“text”列
    if "text" not in df.columns:
        st.error("CSV文件需要包含名为“text”的列。")
    else:
        # 合并所有文本数据
        all_text = " ".join(df["text"].dropna())

        # 创建词云
        wordcloud = WordCloud(width=800, height=800, background_color="white").generate(all_text)

        # 显示词云图
        st.set_option('deprecation.showPyplotGlobalUse', False)  # 防止双图显示
        plt.figure(figsize=(8, 8), facecolor=None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad=0)

        st.pyplot(plt)

def delete_folder(path):
    # 检查文件夹是否存在
    if os.path.exists(path):
        # 使用shutil.rmtree删除文件夹
        shutil.rmtree(path)
        st.write(f"Folder {path} has been deleted.")
    else:
        st.write(f"Folder {path} does not exist.")
    


from sklearn.feature_extraction.text import TfidfVectorizer

def extract_top_tfidf_words(input_list, num_words=50):
    # 创建TF-IDF向量化器
    first_column_data = input_list
   
    with open("liqa/stop.txt", "r", encoding="utf-8") as f:
        chinese_stopwords = [line.strip() for line in f]

    # 创建TF-IDF向量化器，指定停用词为加载的中文停用词表
    tfidf_vectorizer = TfidfVectorizer(stop_words=chinese_stopwords)    

    # 计算TF-IDF矩阵
    tfidf_matrix = tfidf_vectorizer.fit_transform(first_column_data )

    # 获取特征词列表
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # 将TF-IDF矩阵转换为稀疏矩阵
    tfidf_matrix_sparse = tfidf_matrix.toarray()

    # 计算每个词语的平均TF-IDF得分
    avg_tfidf_scores = tfidf_matrix_sparse.mean(axis=0)

    # 创建一个包含词语和对应平均TF-IDF得分的字典
    word_tfidf_scores = {word: score for word, score in zip(feature_names, avg_tfidf_scores)}

    # 根据TF-IDF得分进行降序排序
    sorted_words = sorted(word_tfidf_scores.items(), key=lambda x: x[1], reverse=True)

    # 提取前num_words个高频词和它们的TF-IDF得分
    top_words = sorted_words[:num_words]

    return top_words

def display_local_pdf(pdf_path, ui_width):
    # Read file as bytes:
    with open(pdf_path, 'rb') as pdf_file:
        bytes_data = pdf_file.read()

    # Convert to base64
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')

    # Embed PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width={str(ui_width)} height={str(ui_width*4/3)} type="application/pdf"></iframe>'

    # Display file
    st.markdown(pdf_display, unsafe_allow_html=True)



def count_files_in_folder(folder_path):
    file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    return file_count



def calculate_total_word_count(input_list):
    total_word_count = 0

    for text in input_list:
        words = text.split()  # 使用空格分割单词
        total_word_count += len(words)

    return total_word_count


dataset_list = [item for item in os.listdir('liqa/dataset') if  '.' not in item]
with st.sidebar:
    dataset_sele=sac.menu([f'{i}' for i in dataset_list], index=0, format_func='title', size='middle', indent=24, open_index=None, open_all=True, return_index=True)
    
select= option_menu(None, ["库信息", "文件内容"], icons=["list-task", 'gear'], 
menu_icon="cast", default_index=0, orientation="horizontal")
if select=='库信息':
    st.divider()
    
    col4, col5, col6 = st.columns(3)
    fold=count_files_in_folder("liqa/dataset"+"/"+dataset_list[dataset_sele]+"/"+"source")
    
    if st.sidebar.button('删除此库',use_container_width=True):
        delete_folder("liqa/dataset"+"/"+dataset_list[dataset_sele])
    path_info_json="liqa/dataset"+"/"+dataset_list[dataset_sele]+'/'+'storage'+"/"+"docstore.json"
    import json
    with open(path_info_json, 'r') as f:
        info_json = json.load(f)
    info_json.get('docstore/data')
    info_for_csv=[i.get('__data__').get('text') for i in info_json.get('docstore/data').values()]
    
    sd=len(info_for_csv)
    bj=calculate_total_word_count(info_for_csv)
    col4.metric(label="文件数", value=fold, delta=fold)
    col5.metric(label="文本块", value=sd, delta=sd)
    col6.metric(label="总字数", value=bj, delta=bj)
    style_metric_cards()
    
    a,b=st.columns([3,1])
    
    data = [
        {"name": name, "value": value}
        for name, value in extract_top_tfidf_words(info_for_csv)]
    wordcloud_option = {"series": [{"type": "wordCloud", "data": data}]}
    
    with a:
        st.header('词云图')
        st_echarts(wordcloud_option)
        
    with b:
        st.header('属性表')
        options = {
            "tooltip": {"trigger": "item"},
            "legend": {"top": "5%", "left": "center"},
            "series": [
                {
                    "name": "信息来源",
                    "type": "pie",
                    "radius": ["40%", "70%"],
                    "avoidLabelOverlap": False,
                    "itemStyle": {
                        "borderRadius": 10,
                        "borderColor": "#fff",
                        "borderWidth": 2,
                    },
                    "label": {"show": False, "position": "center"},
                    "emphasis": {
                        "label": {"show": True, "fontSize": "40", "fontWeight": "bold"}
                    },
                    "labelLine": {"show": False},
                    "data": [
                        {"value": 1300, "name": "原文"},
                        {"value": 400, "name": "匹配问题"},
                        {"value": 300, "name": "文章摘要"},
                
                    ],
                }
            ],
        }
        st_echarts(
            options=options, height="500px",
        )
    
    
    
    
    file_list = [item for item in get_filenames_without_extension('liqa/dataset/'+dataset_list[dataset_sele]+'/source') ]
    st.divider()
else:
    a,b=st.columns([1,2],gap="small")
    file_list = [item for item in get_filenames_without_extension('liqa/dataset/'+dataset_list[dataset_sele]+'/source') ]
    with a:
        file_sele=sac.menu(
            [f'{i}' for i in file_list], index=0, format_func='title', size='middle', indent=24, open_index=None, open_all=True, return_index=True)
    with b:
        st.header('源文件')
        ui_width = st_javascript("window.innerWidth")
        
        display_local_pdf("liqa/dataset"+"/"+dataset_list[dataset_sele]+"/"+"source"+"/"+file_list[file_sele]+".pdf",ui_width)

    

        