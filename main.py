import pandas as pd
import streamlit as st
from utils import dataframe_agent


def create_chart(input_data, chart_type):
    df_data = pd.DataFrame(input_data["data"], columns=input_data["columns"])
    df_data.set_index(input_data["columns"][0], inplace=True)
    if chart_type == "bar":
        st.bar_chart(df_data)
    elif chart_type == "line":
        st.line_chart(df_data)
    elif chart_type == "scatter":
        st.scatter_chart(df_data)


st.title("💡 CSV數據分析AI工具")

with st.sidebar:
    openai_api_key = st.text_input("請輸入OpenAI API密鑰：", type="password")
    st.markdown(
        "[獲取OpenAI API key](https://platform.openai.com/account/api-keys)")

data = st.file_uploader("上傳你的數據文件（CSV格式）：", type="csv")
if data:
    st.session_state["df"] = pd.read_csv(data)
    with st.expander("原始數據"):
        st.dataframe(st.session_state["df"])

query = st.text_area("請輸入你關於以上表格的問題，或數據提取請求，或可視化要求（支持散點圖、折線圖、長條圖）：")
button = st.button("生成回答")

if button and not openai_api_key:
    st.info("請輸入你的OpenAI API密鑰")
if button and "df" not in st.session_state:
    st.info("請先上傳數據文件")
if button and openai_api_key and "df" in st.session_state:
    with st.spinner("AI正在思考中，請稍等..."):
        response_dict = dataframe_agent(
            openai_api_key, st.session_state["df"], query)
        if "answer" in response_dict:
            st.write("AI 回答:", response_dict["answer"])
        if "table" in response_dict:
            st.table(pd.DataFrame(response_dict["table"]["data"],
                                  columns=response_dict["table"]["columns"]))
        if "bar" in response_dict:
            create_chart(response_dict["bar"], "bar")
        if "line" in response_dict:
            create_chart(response_dict["line"], "line")
        if "scatter" in response_dict:
            create_chart(response_dict["scatter"], "scatter")
