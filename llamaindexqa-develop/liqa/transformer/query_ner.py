import os

# 将 TMPDIR 环境变量设为当前目录
os.environ['TMPDIR'] = os.getcwd()+'/tokenizer'
os.environ['TIKTOKEN_CACHE_DIR'] = os.getcwd()+'/tokenizer'
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
os.environ['TRANSFORMERS_CACHE'] = '/nvme/share/share/yangyihe/embedding'  # 它会要求这个目录下面的models目录，还强制找models目录，我在这个embeddings下又建设了一个models目录
os.environ["LLAMA_INDEX_CACHE_DIR"] = os.getcwd()+'/tokenizer' # 暂时未知真的有用？ 但偶尔有用
from promptify import Prompter,OpenAI, Pipeline
import openai

sentence     =  """The patient is a 93-year-old female with a medical  				 
                history of chronic right hip pain, osteoporosis,					
                hypertension, depression, and chronic atrial						
                fibrillation admitted for evaluation and management				
                of severe nausea and vomiting and urinary tract				
                infection"""
openai.api_base='https://gf.nekoapi.com/v1'
model        = OpenAI(api_key='sk-aJzbu0F3j7bstWlR3e4cA9Db59Ac4f669a9f471aFa66C458') # or `HubModel()` for Huggingface-based inference or 'Azure' etc
prompter     = Prompter('ner.jinja') # select a template or provide custom template
pipe         = Pipeline(prompter , model)


result = pipe.fit(sentence, domain="medical", labels=None)
print(result)