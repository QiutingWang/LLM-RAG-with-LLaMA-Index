from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType



DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "我们需要完成对话任务，\n"
    "给定的上下文信息如下。\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "我是金融资本市场的从业者\n，"
    "你作为一个人工智能助理，希望使用给定的上下文信息而不使用先验知识， "
    "通过一步一步地思考，准确、详细、全面地回答下面的问题。\n"
    # "回答示例如下。\n"
    # "上下文信息:公开发行的证券，应当在依法设立的证券交易所上市交易或者在国务院批准的其他全国性证券交易场所交易。非公开发行的证券，可以在证券交易所、国务院批准的其他全国性证券交易场所、按照国务院规定设立的区域性股权市场转让。\n"
    # "问题:非公开发行的证券可以在哪里交易？\n"
    # "回答:非公开发行的证券，可以在证券交易所、国务院批准的其他全国性证券交易场所、按照国务院规定设立的区域性股权市场转让。\n"
    # "问题:今天天气如何？\n"
    # "回答:十分抱歉！您的提问超出了我的知识范围，您还可以询问其他合规相关问题，我会尽力为您解答。"
    "问题: {query_str}\n"
    "回答: "


)
DEFAULT_TEXT_QA_PROMPT = PromptTemplate(
    DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)

DEFAULT_REFINE_PROMPT_TMPL = (
    "原始问题如下: {query_str}\n"
    "我们已经提供了一个现有的答案: {existing_answer}\n"
    "(仅在需要时)参照如下更多的上下文，我们有机会改善现有的答案"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "参照新的上下文，将原有的答案回答得更全面、准确，\n"
    "通过一步一步地思考并准确地回答原始问题。\n"
    "如果新的上下文没有用处，请返回原有的答案。\n"
    "完善后的答案: "
)

DEFAULT_REFINE_PROMPT = PromptTemplate(
    DEFAULT_REFINE_PROMPT_TMPL, prompt_type=PromptType.REFINE
)



# DEFAULT_SUMMARY_PROMPT_TMPL = (
#     "Write a summary of the following. Try to use only the "
#     "information provided. "
#     "Try to include as many key details as possible.\n"
#     "\n"
#     "\n"
#     "{context_str}\n"
#     "\n"
#     "\n"
#     'SUMMARY:"""\n'
# )

# DEFAULT_SUMMARY_PROMPT = PromptTemplate(
#     DEFAULT_SUMMARY_PROMPT_TMPL, prompt_type=PromptType.SUMMARY
# )

# # insert prompts
# DEFAULT_INSERT_PROMPT_TMPL = (
#     "Context information is below. It is provided in a numbered list "
#     "(1 to {num_chunks}), "
#     "where each item in the list corresponds to a summary.\n"
#     "---------------------\n"
#     "{context_list}"
#     "---------------------\n"
#     "Given the context information, here is a new piece of "
#     "information: {new_chunk_text}\n"
#     "Answer with the number corresponding to the summary that should be updated. "
#     "The answer should be the number corresponding to the "
#     "summary that is most relevant to the question.\n"
# )
# DEFAULT_INSERT_PROMPT = PromptTemplate(
#     DEFAULT_INSERT_PROMPT_TMPL, prompt_type=PromptType.TREE_INSERT
# )


# # # single choice
# DEFAULT_QUERY_PROMPT_TMPL = (
#     "Some choices are given below. It is provided in a numbered list "
#     "(1 to {num_chunks}), "
#     "where each item in the list corresponds to a summary.\n"
#     "---------------------\n"
#     "{context_list}"
#     "\n---------------------\n"
#     "Using only the choices above and not prior knowledge, return "
#     "the choice that is most relevant to the question: '{query_str}'\n"
#     "Provide choice in the following format: 'ANSWER: <number>' and explain why "
#     "this summary was selected in relation to the question.\n"
# )
# DEFAULT_QUERY_PROMPT = PromptTemplate(
#     DEFAULT_QUERY_PROMPT_TMPL, prompt_type=PromptType.TREE_SELECT
# )

# # multiple choice
# DEFAULT_QUERY_PROMPT_MULTIPLE_TMPL = (
#     "Some choices are given below. It is provided in a numbered "
#     "list (1 to {num_chunks}), "
#     "where each item in the list corresponds to a summary.\n"
#     "---------------------\n"
#     "{context_list}"
#     "\n---------------------\n"
#     "Using only the choices above and not prior knowledge, return the top choices "
#     "(no more than {branching_factor}, ranked by most relevant to least) that "
#     "are most relevant to the question: '{query_str}'\n"
#     "Provide choices in the following format: 'ANSWER: <numbers>' and explain why "
#     "these summaries were selected in relation to the question.\n"
# )
# DEFAULT_QUERY_PROMPT_MULTIPLE = PromptTemplate(
#     DEFAULT_QUERY_PROMPT_MULTIPLE_TMPL, prompt_type=PromptType.TREE_SELECT_MULTIPLE
# )

# DEFAULT_TREE_SUMMARIZE_TMPL = (
#     "Context information from multiple sources is below.\n"
#     "---------------------\n"
#     "{context_str}\n"
#     "---------------------\n"
#     "Given the information from multiple sources and not prior knowledge, "
#     "answer the query.\n"
#     "Query: {query_str}\n"
#     "Answer: "
# )
# DEFAULT_TREE_SUMMARIZE_PROMPT = PromptTemplate(
#     DEFAULT_TREE_SUMMARIZE_TMPL, prompt_type=PromptType.SUMMARY
# )


# ############################################
# # Keyword Table
# ############################################

# DEFAULT_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
#     "Some text is provided below. Given the text, extract up to {max_keywords} "
#     "keywords from the text. Avoid stopwords."
#     "---------------------\n"
#     "{text}\n"
#     "---------------------\n"
#     "Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'\n"
# )
# DEFAULT_KEYWORD_EXTRACT_TEMPLATE = PromptTemplate(
#     DEFAULT_KEYWORD_EXTRACT_TEMPLATE_TMPL, prompt_type=PromptType.KEYWORD_EXTRACT
# )


# # NOTE: the keyword extraction for queries can be the same as
# # the one used to build the index, but here we tune it to see if performance is better.
# DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL = (
#     "A question is provided below. Given the question, extract up to {max_keywords} "
#     "keywords from the text. Focus on extracting the keywords that we can use "
#     "to best lookup answers to the question. Avoid stopwords.\n"
#     "---------------------\n"
#     "{question}\n"
#     "---------------------\n"
#     "Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'\n"
# )
# DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE = PromptTemplate(
#     DEFAULT_QUERY_KEYWORD_EXTRACT_TEMPLATE_TMPL,
#     prompt_type=PromptType.QUERY_KEYWORD_EXTRACT,
# )


# ############################################
# # Structured Store
# ############################################

# DEFAULT_SCHEMA_EXTRACT_TMPL = (
#     "We wish to extract relevant fields from an unstructured text chunk into "
#     "a structured schema. We first provide the unstructured text, and then "
#     "we provide the schema that we wish to extract. "
#     "-----------text-----------\n"
#     "{text}\n"
#     "-----------schema-----------\n"
#     "{schema}\n"
#     "---------------------\n"
#     "Given the text and schema, extract the relevant fields from the text in "
#     "the following format: "
#     "field1: <value>\nfield2: <value>\n...\n\n"
#     "If a field is not present in the text, don't include it in the output."
#     "If no fields are present in the text, return a blank string.\n"
#     "Fields: "
# )
# DEFAULT_SCHEMA_EXTRACT_PROMPT = PromptTemplate(
#     DEFAULT_SCHEMA_EXTRACT_TMPL, prompt_type=PromptType.SCHEMA_EXTRACT
# )

# # NOTE: taken from langchain and adapted
# # https://github.com/langchain-ai/langchain/blob/v0.0.303/libs/langchain/langchain/chains/sql_database/prompt.py
# DEFAULT_TEXT_TO_SQL_TMPL = (
#     "Given an input question, first create a syntactically correct {dialect} "
#     "query to run, then look at the results of the query and return the answer. "
#     "You can order the results by a relevant column to return the most "
#     "interesting examples in the database.\n\n"
#     "Never query for all the columns from a specific table, only ask for a "
#     "few relevant columns given the question.\n\n"
#     "Pay attention to use only the column names that you can see in the schema "
#     "description. "
#     "Be careful to not query for columns that do not exist. "
#     "Pay attention to which column is in which table. "
#     "Also, qualify column names with the table name when needed. "
#     "You are required to use the following format, each taking one line:\n\n"
#     "Question: Question here\n"
#     "SQLQuery: SQL Query to run\n"
#     "SQLResult: Result of the SQLQuery\n"
#     "Answer: Final answer here\n\n"
#     "Only use tables listed below.\n"
#     "{schema}\n\n"
#     "Question: {query_str}\n"
#     "SQLQuery: "
# )

# DEFAULT_TEXT_TO_SQL_PROMPT = PromptTemplate(
#     DEFAULT_TEXT_TO_SQL_TMPL,
#     prompt_type=PromptType.TEXT_TO_SQL,
# )

# DEFAULT_TEXT_TO_SQL_PGVECTOR_TMPL = """\
# Given an input question, first create a syntactically correct {dialect} \
# query to run, then look at the results of the query and return the answer. \
# You can order the results by a relevant column to return the most \
# interesting examples in the database.

# Pay attention to use only the column names that you can see in the schema \
# description. Be careful to not query for columns that do not exist. \
# Pay attention to which column is in which table. Also, qualify column names \
# with the table name when needed.

# IMPORTANT NOTE: you can use specialized pgvector syntax (`<->`) to do nearest \
# neighbors/semantic search to a given vector from an embeddings column in the table. \
# The embeddings value for a given row typically represents the semantic meaning of that row. \
# The vector represents an embedding representation \
# of the question, given below. Do NOT fill in the vector values directly, but rather specify a \
# `[query_vector]` placeholder. For instance, some select statement examples below \
# (the name of the embeddings column is `embedding`):
# SELECT * FROM items ORDER BY embedding <-> '[query_vector]' LIMIT 5;
# SELECT * FROM items WHERE id != 1 ORDER BY embedding <-> (SELECT embedding FROM items WHERE id = 1) LIMIT 5;
# SELECT * FROM items WHERE embedding <-> '[query_vector]' < 5;

# You are required to use the following format, \
# each taking one line:

# Question: Question here
# SQLQuery: SQL Query to run
# SQLResult: Result of the SQLQuery
# Answer: Final answer here

# Only use tables listed below.
# {schema}


# Question: {query_str}
# SQLQuery: \
# """

# DEFAULT_TEXT_TO_SQL_PGVECTOR_PROMPT = PromptTemplate(
#     DEFAULT_TEXT_TO_SQL_PGVECTOR_TMPL,
#     prompt_type=PromptType.TEXT_TO_SQL,
# )


# # NOTE: by partially filling schema, we can reduce to a QuestionAnswer prompt
# # that we can feed to ur table
# DEFAULT_TABLE_CONTEXT_TMPL = (
#     "We have provided a table schema below. "
#     "---------------------\n"
#     "{schema}\n"
#     "---------------------\n"
#     "We have also provided context information below. "
#     "{context_str}\n"
#     "---------------------\n"
#     "Given the context information and the table schema, "
#     "give a response to the following task: {query_str}"
# )

# DEFAULT_TABLE_CONTEXT_QUERY = (
#     "Provide a high-level description of the table, "
#     "as well as a description of each column in the table. "
#     "Provide answers in the following format:\n"
#     "TableDescription: <description>\n"
#     "Column1Description: <description>\n"
#     "Column2Description: <description>\n"
#     "...\n\n"
# )

# DEFAULT_TABLE_CONTEXT_PROMPT = PromptTemplate(
#     DEFAULT_TABLE_CONTEXT_TMPL, prompt_type=PromptType.TABLE_CONTEXT
# )

# # NOTE: by partially filling schema, we can reduce to a refine prompt
# # that we can feed to ur table
# DEFAULT_REFINE_TABLE_CONTEXT_TMPL = (
#     "We have provided a table schema below. "
#     "---------------------\n"
#     "{schema}\n"
#     "---------------------\n"
#     "We have also provided some context information below. "
#     "{context_msg}\n"
#     "---------------------\n"
#     "Given the context information and the table schema, "
#     "give a response to the following task: {query_str}\n"
#     "We have provided an existing answer: {existing_answer}\n"
#     "Given the new context, refine the original answer to better "
#     "answer the question. "
#     "If the context isn't useful, return the original answer."
# )
# DEFAULT_REFINE_TABLE_CONTEXT_PROMPT = PromptTemplate(
#     DEFAULT_REFINE_TABLE_CONTEXT_TMPL, prompt_type=PromptType.TABLE_CONTEXT
# )


# ############################################
# # Knowledge-Graph Table
# ############################################

# DEFAULT_KG_TRIPLET_EXTRACT_TMPL = (
#     "Some text is provided below. Given the text, extract up to "
#     "{max_knowledge_triplets} "
#     "knowledge triplets in the form of (subject, predicate, object). Avoid stopwords.\n"
#     "---------------------\n"
#     "Example:"
#     "Text: Alice is Bob's mother."
#     "Triplets:\n(Alice, is mother of, Bob)\n"
#     "Text: Philz is a coffee shop founded in Berkeley in 1982.\n"
#     "Triplets:\n"
#     "(Philz, is, coffee shop)\n"
#     "(Philz, founded in, Berkeley)\n"
#     "(Philz, founded in, 1982)\n"
#     "---------------------\n"
#     "Text: {text}\n"
#     "Triplets:\n"
# )
# DEFAULT_KG_TRIPLET_EXTRACT_PROMPT = PromptTemplate(
#     DEFAULT_KG_TRIPLET_EXTRACT_TMPL, prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT
# )

# ############################################
# # HYDE
# ##############################################

# HYDE_TMPL = (
#     "Please write a passage to answer the question\n"
#     "Try to include as many key details as possible.\n"
#     "\n"
#     "\n"
#     "{context_str}\n"
#     "\n"
#     "\n"
#     'Passage:"""\n'
# )

# DEFAULT_HYDE_PROMPT = PromptTemplate(HYDE_TMPL, prompt_type=PromptType.SUMMARY)


# ############################################
# # Simple Input
# ############################################

# DEFAULT_SIMPLE_INPUT_TMPL = "{query_str}"
# DEFAULT_SIMPLE_INPUT_PROMPT = PromptTemplate(
#     DEFAULT_SIMPLE_INPUT_TMPL, prompt_type=PromptType.SIMPLE_INPUT
# )


# ############################################
# # Pandas
# ############################################

# DEFAULT_PANDAS_TMPL = (
#     "You are working with a pandas dataframe in Python.\n"
#     "The name of the dataframe is `df`.\n"
#     "This is the result of `print(df.head())`:\n"
#     "{df_str}\n\n"
#     "Here is the input query: {query_str}.\n"
#     "Given the df information and the input query, please follow "
#     "these instructions:\n"
#     "{instruction_str}"
#     "Output:\n"
# )

# DEFAULT_PANDAS_PROMPT = PromptTemplate(
#     DEFAULT_PANDAS_TMPL, prompt_type=PromptType.PANDAS
# )


# ############################################
# # JSON Path
# ############################################

# DEFAULT_JSON_PATH_TMPL = (
#     "We have provided a JSON schema below:\n"
#     "{schema}\n"
#     "Given a task, respond with a JSON Path query that "
#     "can retrieve data from a JSON value that matches the schema.\n"
#     "Task: {query_str}\n"
#     "JSONPath: "
# )

# DEFAULT_JSON_PATH_PROMPT = PromptTemplate(
#     DEFAULT_JSON_PATH_TMPL, prompt_type=PromptType.JSON_PATH
# )


# ############################################
# # Choice Select
# ############################################

# DEFAULT_CHOICE_SELECT_PROMPT_TMPL = (
#     "A list of documents is shown below. Each document has a number next to it along "
#     "with a summary of the document. A question is also provided. \n"
#     "Respond with the numbers of the documents "
#     "you should consult to answer the question, in order of relevance, as well \n"
#     "as the relevance score. The relevance score is a number from 1-10 based on "
#     "how relevant you think the document is to the question.\n"
#     "Do not include any documents that are not relevant to the question. \n"
#     "Example format: \n"
#     "Document 1:\n<summary of document 1>\n\n"
#     "Document 2:\n<summary of document 2>\n\n"
#     "...\n\n"
#     "Document 10:\n<summary of document 10>\n\n"
#     "Question: <question>\n"
#     "Answer:\n"
#     "Doc: 9, Relevance: 7\n"
#     "Doc: 3, Relevance: 4\n"
#     "Doc: 7, Relevance: 3\n\n"
#     "Let's try this now: \n\n"
#     "{context_str}\n"
#     "Question: {query_str}\n"
#     "Answer:\n"
# )
# DEFAULT_CHOICE_SELECT_PROMPT = PromptTemplate(
#     DEFAULT_CHOICE_SELECT_PROMPT_TMPL, prompt_type=PromptType.CHOICE_SELECT
# )