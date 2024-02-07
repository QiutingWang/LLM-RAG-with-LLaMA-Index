import os
from config import config
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
import streamlit_antd_components as sac
import openai
from utils import retriver_prompt,markdown_insert_images
import pandas as pd
from dataclasses import dataclass, asdict
import streamlit_antd_components as sac
from streamlit_antd_components.utils.data_class import BsIcon
from llama_index import (
            StorageContext,  #定义了存储文档、嵌入和索引的存储后端
            load_index_from_storage)

from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType

# 删除streamlit的页脚，隐藏右上角菜单栏
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

#用st实现页面布局
with st.sidebar:
    st.image('liqa/images/红.png')
    
    selected2 = option_menu(None, [ "问答", "配置项"],
                                icons=['gear', 'cloud-upload',] ,
                                menu_icon="cast", default_index=0)



# 除了侧边栏的部分分为两页，一页展示数据，一页展示交互
# 交互部分分为两栏，左栏展示问题，右栏展示答案

def on_btn_click():
    del st.session_state.messages
    del st.session_state.messages_latent

@st.cache_resource
def load_index(selected_file):
    # 加载索引的代码
     # if selected_file:
    storage_context = StorageContext.from_defaults(persist_dir="liqa/dataset"+"/"+selected_file[0]+'/'+'storage')
    # load index
    index = load_index_from_storage(storage_context)  
    return index


if selected2 == '问答':
    #载入可选的数据库，单选，先查询本地dataset文件夹下的文件目录
    # 读取文件夹下的文件名
    file_list = [item for item in os.listdir('liqa/dataset') if  '.' not in item]
    #查询当前路径下的文件夹都有哪些
    
    with st.sidebar:    
        st.selectbox('向量引擎', config.EMBEDDING_NAME,index=0)
        st.session_state["openai_model"]=st.selectbox('问答助手',config.MODEL_NAME)
        
        st.divider()
        
        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = None
        if "e_model" not in st.session_state:
            st.session_state.e_model = None
        if "c_model" not in st.session_state:
                    st.session_state.c_model = None
        if st.session_state['c_model']:
            st.sidebar.success(f"{st.session_state['c_model']}运行中", icon="ℹ️")
        st.button("清空对话", on_click=on_btn_click,use_container_width=True) 
        st.button("导出对话",use_container_width=True)  
    user_avator = "liqa/images/user.png"
    assistant_avator = "liqa/images/robot.png"
    ## TODO初始化对话配置
    st.session_state.max_length=2048
    st.session_state.top_p=  0.8
    st.session_state.temperature = 0.7
    a,c=st.columns([1,3])
    
    with a:
        a1,c1=st.columns([1,1])
        with a1:
            sac_web=sac.switch(label='**联网功能**', value=False, checked=BsIcon(name='wifi'), unchecked=BsIcon(name='wifi-off'), align='center', position='top', size='large', disabled=True)
        with c1:
            sac_memory=sac.switch(label='**记忆**', value=False, checked=BsIcon(name='memory'), unchecked=BsIcon(name='x'), align='center', position='top', size='large', disabled=True)
   
    with c:
        a,b,c=st.columns([1,3,1])
        with a:
            sac_file=sac.switch(label='**数据库功能**', value=True, checked=BsIcon(name='database-fill-check'), unchecked=BsIcon(name='database-fill-slash'), align='center', position='top', size='large', disabled=False)
        with b:
            selected_file = st.multiselect('**选择要对话的数据库**',file_list,disabled=not sac_file)
            if selected_file:
                index=load_index(selected_file)
        with c:
             K_text_chunks=int(st.number_input('**片段数量**',value=3,disabled=not sac_file))
    st.title("对话助手")


    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": '你好，有什么可以帮您的'}]
    if "messages_latent" not in st.session_state:
        st.session_state.messages_latent = [{"role": "assistant", "content": '你好，有什么可以帮您的'}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"],avatar=assistant_avator):
            st.markdown(message["content"])

    if prompt := st.chat_input("想咨询什么?"):
        # 如果开启数据库查询式问答
        with st.chat_message("user",avatar=user_avator):
            st.markdown(markdown_insert_images(prompt),unsafe_allow_html=True)
            st.session_state.messages.append({"role": "user", "content": prompt})
            if sac_file:
                print(sac_file,"sac_file当前状态")
                
               
            st.session_state.messages_latent.append({"role": "user", "content": prompt})

        with st.chat_message("assistant",avatar=assistant_avator):
            
            message_placeholder = st.empty()
           
            full_response = ""
            openai.api_key =config.API_KEYS
            
            
            TEXT_QA_PROMPT_TMPL = (
                "以下是上下文信息。\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "根据上述上下文信息，而不是先验知识，回答以下查询。\n"
                "查询：{query_str}\n"
                "回答： "
            )
            TEXT_QA_PROMPT = PromptTemplate(
                TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
            )
            
            
            REFINE_PROMPT_TMPL = (
                "原始查询如下：{query_str}\n"
                "我们已经提供了一个现有的答案：{existing_answer}\n"
                "我们有机会（如果需要的话）使用下面的更多上下文来优化现有的答案。\n"
                "------------\n"
                "{context_msg}\n"
                "------------\n"
                "根据新的上下文，优化原始答案以更好地回答查询。"
                "如果上下文并无帮助，返回原始答案。\n"
                "优化后的答案： "
            )
            REFINE_PROMPT = PromptTemplate(
                REFINE_PROMPT_TMPL, prompt_type=PromptType.REFINE
            )
                        
            openai.api_base =config.API_BASE
            query_engine = index.as_query_engine(
                text_qa_template=TEXT_QA_PROMPT,
                refine_template=REFINE_PROMPT ,
                streaming=True,
                similarity_top_k=K_text_chunks
            )
            responses=query_engine.query(st.session_state.messages_latent[-1]['content'])
            for response in responses.response_gen:
                full_response += response
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(markdown_insert_images(full_response),unsafe_allow_html=True)
           
            with st.expander("参考展示"):
                st.write(responses.get_formatted_sources())
                        
         
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.messages_latent.append({"role": "assistant", "content": full_response})
        rating = st.radio("打分回复:",help='help', options=[1, 2, 3], index=1,horizontal=True)



if selected2=='配置项':
    with st.expander("对话配置",expanded=True):
        max_length = st.slider("Max Length", min_value=32, max_value=2048, value=2048)
        top_p = st.slider(
            'Top P', 0.0, 1.0, 0.8, step=0.01
        )
        temperature = st.slider(
            'Temperature', 0.0, 1.0, 0.7, step=0.01
        )