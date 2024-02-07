from rank_bm25 import BM25Okapi
import jieba

# 示例数据（中文文本）
documents = [
    "这是BM25搜索算法的一个例子",
    "另一个BM25算法的例子",
    "Python编程很有趣",
    "BM25算法用于文本检索任务"
]

# 分词处理
def chinese_tokenize(text):
    return list(jieba.cut(text))

documents_tokenized = [chinese_tokenize(doc) for doc in documents]

# 构建BM25模型
bm25 = BM25Okapi(documents_tokenized)

# 查询词
query = "搜索算法"

# 分词处理
query_tokenized = chinese_tokenize(query)
print(query_tokenized)
# 获取文档的BM25分数
scores = bm25.get_scores(query_tokenized)

# 输出文档的BM25分数
print("BM25 Scores:", scores)