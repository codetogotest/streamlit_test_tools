import json
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent


PROMPT_TEMPLATE = """
你是一位數據分析助理，你的回應內容取決於用戶的請求內容。

1. 對於文字回答的問題，按照這樣的格式回答：
   {"answer": "<你的答案寫在這裡>"}
例如：
   {"answer": "訂單量最高的產品ID是'MD3-76'"}

2. 如果用戶需要一個表格，按照這樣的格式回答：
   {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

3. 如果用戶的請求適合返回長形圖，按照這樣的格式回答：
   {"bar": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}

4. 如果用戶的請求適合返回折線圖，按照這樣的格式回答：
   {"line": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}

5. 如果用戶的請求適合返回散點圖，按照這樣的格式回答：
   {"scatter": {"columns": ["A", "B", "C", ...], "data": [34, 21, 91, ...]}}
注意：我們只支持三種類型的圖表："bar", "line" 和 "scatter"。


請將所有輸出作為JSON字符串返回。請注意要將"columns"列表和數據列表中的所有字符串都用雙引號包圍。
例如：{"columns": ["Products", "Orders"], "data": [["32085Lip", 245], ["76439Eye", 178]]}

你要處理的用戶請求如下： 
"""


def dataframe_agent(openai_api_key, df, query):
    model = ChatOpenAI(model="gpt-4-turbo",
                       openai_api_key=openai_api_key,
                       temperature=0)
    agent = create_pandas_dataframe_agent(llm=model,
                                          df=df,
                                          agent_executor_kwargs={

                                              "handle_parsing_errors": True},
                                          allow_dangerous_code=True,
                                          verbose=True)
    prompt = PROMPT_TEMPLATE + query

    response = agent.invoke({"input": prompt})

    response_dict = json.loads(response["output"])
    return response_dict
