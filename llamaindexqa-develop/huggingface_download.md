# 模型下载指南-无废话版

```
pip install -U "huggingface_hub[cli]"
```

## 可能会有微调模型，所以不用原生的结构了，自己维护一套把

download_path='/nvme/share/share/yangyihe'

>>> 有效的镜像:https://hf-mirror.com/

扫描缓存：想知道下载了哪些存储库以及它在磁盘上占用了多少空间

>>> huggingface-cli scan-cache --dir .

删除缓存：帮助您删除不再使用的缓存部分

>>> huggingface-cli delete-cache

下载示例

>>> huggingface-cli download --cache-dir ./embedding thenlper/gte-large-zh


