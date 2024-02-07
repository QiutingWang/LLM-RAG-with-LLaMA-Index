# source /nvme/share/share/yangyihe/env/lib/python3.10/venv/scripts/common/activate
# vllm的启动脚本
# huggingface-cli scan-cache  --dir .
#    --model THUDM/chatglm3-6b \
#    --model Qwen/Qwen-14B-Chat \

# sh vllm_script/vllm_start.sh
# sh vllm_script/vllm_stop.sh
#!/bin/bash

# # 启动第一个模型服务并且记录它的 PID
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
       --port 8000 \
       --model THUDM/chatglm3-6b \
       --trust-remote-code \
       --download-dir /nvme/share/share/yangyihe/models \
       --tensor-parallel-size 1 \
       --dtype float16 \
       --gpu-memory-utilization 0.9 \
       --tokenizer-mode auto  > /tmp/model1.log 2>&1 & echo $! > /tmp/model1.pid

# 启动第二个模型服务并且记录它的 PID
CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server \
       --port 8001 \
       --model internlm/internlm-chat-7b-8k \
       --trust-remote-code \
       --download-dir /nvme/share/share/yangyihe/models \
       --tensor-parallel-size 1 \
       --dtype float16 \
       --gpu-memory-utilization 1 \
       --tokenizer-mode auto  > /tmp/model2.log 2>&1 & echo $! > /tmp/model2.pid

# 启动第三个模型服务并且记录它的 PID
CUDA_VISIBLE_DEVICES=2,3,4,5 nohup python -m vllm.entrypoints.openai.api_server \
       --port 8002 \
       --model Qwen/Qwen-14B-Chat \
       --trust-remote-code \
       --download-dir /nvme/share/share/yangyihe/models \
       --tensor-parallel-size 4 \
       --dtype float16 \
       --gpu-memory-utilization 0.9 \
       --tokenizer-mode auto  > /tmp/model3.log 2>&1 & echo $! > /tmp/model3.pid

# # 启动第四个模型服务并且记录它的 PID 20B本身存在问题，暂时先不启动
# CUDA_VISIBLE_DEVICES=4,5 nohup python -m vllm.entrypoints.openai.api_server \
#        --port 8003 \
#        --model internlm/internlm-chat-20b \
#        --trust-remote-code \
#        --download-dir /nvme/share/share/yangyihe/models \
#        --tensor-parallel-size 4 \
#        --dtype half \
#        --gpu-memory-utilization 0.9 \
#        --tokenizer-mode auto  > /tmp/model4.log 2>&1 & echo $! > /tmp/model4.pid
