from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
from langchain_classic.agents import initialize_agent, AgentType
# Tools
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun # Use to search over the internet
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper

import streamlit as st
from langchain_classic.callbacks import StreamlitCallbackHandler

# Arxiv tool
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# Wikipedia tool
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# Search Engine tool
search = DuckDuckGoSearchRun(name="Search")

# Combined tool
tools = [arxiv, wiki, search]


st.title("LangChain - Chat with Search")

# Sidebar
st.sidebar.title("Settings")
groq_api_key = st.sidebar.text_input(label="Enter your Groq API Key:", type="password")

# Chat History
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi! I am a chatbot who can search the web. How can I help you?"
        }
    ]

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# Prompt
if (prompt:=st.chat_input(placeholder="What is Machine Learning?")) and groq_api_key:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile", streaming=True)
    search_agent = initialize_agent(llm=llm, 
                                    tools=tools, 
                                    prompt=prompt, 
                                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                    handling_parsing_error=True)
    
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
