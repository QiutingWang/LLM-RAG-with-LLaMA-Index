#!/bin/bash

# 停止第一个模型服务
if [ -f /tmp/model1.pid ]; then
    kill $(cat /tmp/model1.pid)
    rm /tmp/model1.pid
fi

# 停止第二个模型服务
if [ -f /tmp/model2.pid ]; then
    kill $(cat /tmp/model2.pid)
    rm /tmp/model2.pid
fi

# 停止第三个模型服务
if [ -f /tmp/model3.pid ]; then
    kill $(cat /tmp/model3.pid)
    rm /tmp/model3.pid
fi

# 停止第四个模型服务
if [ -f /tmp/model4.pid ]; then
    kill $(cat /tmp/model4.pid)
    rm /tmp/model4.pid
fi