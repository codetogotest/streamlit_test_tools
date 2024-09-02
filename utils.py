import json  # Importing the json module for working with JSON data
import os  # Importing the os module for interacting with the operating system
import pandas as pd  # Importing the pandas library for data manipulation and analysis
# Importing the ChatOpenAI class from the langchain_openai module
from langchain_openai import ChatOpenAI
# Importing the create_pandas_dataframe_agent function from the langchain_experimental.agents.agent_toolkits module
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# Importing the load_dotenv function from the dotenv module
from dotenv import load_dotenv
from openai import Client  # Importing the Client class from the openai module


load_dotenv()  # Loading environment variables from a .env file
# Getting the value of the "OPENAI_API_KEY" environment variable
open_api_key = os.environ.get("OPENAI_API_KEY")
# Getting the value of the "LangSmithLANGCHAIN_API_KEY" environment variable
LangSmith_api_key = os.environ.get("LangSmithLANGCHAIN_API_KEY")

PROMPT_TEMPLATE = """
你是一位數據分析助理，你的回應內容取決於用戶的請求內容。

1. 對於文字回答的問題，按照這樣的格式回答：
   {"answer": "<你的答案寫在這裡>"}
例如：
   {"answer": "訂單量最高的產品ID是'MNWC3-067'"}

2. 如果用戶需要一個表格，按照這樣的格式回答：
   {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

3. 如果用戶的請求適合返回條形圖，按照這樣的格式回答：
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
    model = ChatOpenAI(model="gpt-4-turbo",  # Creating an instance of the ChatOpenAI class with the "gpt-4-turbo" model
                       # Passing the OpenAI API key to the ChatOpenAI instance
                       openai_api_key=openai_api_key,
                       temperature=0)  # Setting the temperature parameter to 0 for deterministic responses
    agent = create_pandas_dataframe_agent(llm=model,  # Creating a pandas dataframe agent using the ChatOpenAI instance
                                          df=df,  # Passing the pandas DataFrame to the agent
                                          agent_executor_kwargs={
                                              # Setting the agent executor kwargs to handle parsing errors
                                              "handle_parsing_errors": True},
                                          allow_dangerous_code=True,  # Allowing dangerous code execution in the agent
                                          verbose=True)  # Enabling verbose mode for the agent
    # Constructing the prompt by concatenating the PROMPT_TEMPLATE and the query
    prompt = PROMPT_TEMPLATE + query
    # Invoking the agent with the prompt as input
    response = agent.invoke({"input": prompt})
    # Parsing the agent's response as a JSON string and converting it to a dictionary
    response_dict = json.loads(response["output"])
    return response_dict  # Returning the response dictionary


# # Reading the CSV file into a pandas DataFrame
# df = pd.read_csv("personal_data.csv")
# # Calling the dataframe_agent function and printing the result
# print(dataframe_agent(os.environ["OPENAI_API_KEY"], df, "數據裡面出現最多的職業是什麼？"))
