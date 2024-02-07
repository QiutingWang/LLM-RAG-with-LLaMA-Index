
from typing import Optional, List, Mapping, Any
from llama_index.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.llms.base import llm_completion_callback
from utils.prompter import Prompter
prompter = Prompter()
import requests
import json
import aiohttp
import asyncio

import json
'''
curl http://103.177.28.196:8123/v1/chat/completions\
  -H "Content-Type: application/json" \
  -H "" \
  -d '{
    "model": "Baichuan2-13B-Chat",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'
'''


class InternLLM(CustomLLM):
    sem = asyncio.Semaphore(80) # 信号量，控制协程数，防止爬的过快
    context_window: int = 3900
    num_output: int = 1024
    model_name: str = "custom"
    dummy_response: str = "My response"
    url = "http://103.177.28.196:8123/v1/chat/completions"
    data = {
    "model": "Baichuan2-13B-Chat",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
        ]
         }
    headers = {
            'Content-Type': 'application/json'
        }
    
    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        
        prompt= prompter.generate_prompt('测试',prompt)
        
        response_text=self.give_response(prompt)
        
        return CompletionResponse(text=json.loads(response_text)['choices'][0]['message']['content'])
    
    @llm_completion_callback()
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        prompt= prompter.generate_prompt('测试',prompt)
        response_text = await self.agive_response(prompt)
        return CompletionResponse(text=json.loads(response_text)['choices'][0]['message']['content'])
    
    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        # TODO: 还没做
        response = ""
        for token in self.dummy_response:
            response += token
            yield CompletionResponse(text=response, delta=token)

    def give_response(self,input):
        data=self.data
        data['messages'][-1]['content']=input
        response=requests.post(self.url, data=json.dumps(data), headers=self.headers)
        return response.text

    async def agive_response(self,input):
        sem=self.sem
        async with sem:
            async with aiohttp.ClientSession(trust_env=True) as session:
                '''
                async with aiohttp.ClientSession(connector=con, trust_env=True) as sess。当 trust_env 设置为 True 时，库将尝试从操作系统环境变量（HTTP_PROXY，HTTPS_PROXY等）读取代理设置。如果没有设置或设置为 False，则不会读取任何操作系统环境设置。注意，如果在代码中显式设置了一个代理，这不会影响该代理设置。
                '''
                data = self.data
                data['messages'][-1]['content']=input
                async with session.post(self.url, data=json.dumps(data), headers=self.headers) as response:
                    return await response.text()
if __name__=='__main__':
                
  our_llm = InternLLM() 
  print(our_llm.complete('你好，你是谁？'))

  import asyncio
  import time

  # 假设 our_llm 已经在此之前被定义并初始化

  # 定义一个异步函数来调用OurLLM的acomplete方法，并打印执行时间
  async def main(i):
      start_time = time.time()
      prompt = f"Once upon a time, {i}"
      response = await our_llm.acomplete(prompt)
      end_time = time.time()
      print(f"Task {i} finished in {end_time - start_time} seconds.",response.text)
      
      return response  # 返回acomplete的结果
  start_time = time.time()
  # 创建多个异步任务
  tasks = [main(i) for i in range(150)]

  # 定义一个新的异步函数来运行所有异步任务
  async def run_all_tasks():
      await asyncio.gather(*tasks)

  # 使用asyncio.run()来运行所有异步任务
  asyncio.run(run_all_tasks())
  end_time = time.time()
  print(f"总时间{end_time - start_time} seconds.")