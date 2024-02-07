import os
import fitz  # PyMuPDF
from llama_index import Document
from  pathlib  import Path
def read_pdf_and_create_document(pdf_path):
    # 打开 PDF 文件
    doc = fitz.open(pdf_path)

    text = ""
    for page in doc:
        # 从每一页读取文本
        text += page.get_text()

    # 创建一个带有元数据的文档
    document = Document(
        text=text,
        metadata={"file_name": Path(pdf_path).name},
    )

    return document

def load_docu(input):
    """
    不包含pdf_page的方式加载node
    """
    documents = []

    # 检查输入是否是一个路径
    if isinstance(input, str) and os.path.isdir(input):
        filenames = os.listdir(input)
    # 检查输入是否是一个列表
    elif isinstance(input, list):
        filenames = input
    else:
        raise ValueError("输入必须是一个路径或者一个文件名列表")

    for filename in filenames:
        if filename.endswith(".pdf"):
            document = read_pdf_and_create_document(filename)
            documents.append(document)
    
    return documents