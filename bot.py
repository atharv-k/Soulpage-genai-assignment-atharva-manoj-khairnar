import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_classic.agents import initialize_agent
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_classic import hub
from langchain_classic.agents.agent_types import AgentType

# --- 1. Load Environment Variables ---
load_dotenv()
if "GROQ_API_KEY" not in os.environ:
    st.error("Please set your GROQ_API_KEY in the .env file or environment variables.")
    st.stop()
groq_api_key = os.getenv("GROQ_API_KEY")

# --- 2. Initialize Components ---

# A. Language Model (LLM)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=groq_api_key
) 

# B. Tools (External Search)
# DuckDuckGoSearchRun: the tool wrapper for a free web search.
search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="DuckDuckGoSearch",
        func=search.run,
        description="Useful for when you need to answer questions about current events, factual information, or things you don't know.",
    )
]

# C. Memory Component
# ConversationBufferMemory: stores the history, which the agent uses for context.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- 3. Create the Conversational Agent ---

# A. Agent Prompt Template
# This prompt structure uses the MessagesPlaceholder to inject history from memory 
# and the agent_scratchpad for intermediate thoughts/tool calls.
system_q_prompt = hub.pull("hwchase17/react-chat").template
# Pulling a standard template for tools-enabled conversational agents
prompt = ChatPromptTemplate.from_messages([
    ("system", system_q_prompt),
    MessagesPlaceholder(variable_name="chat_history"), # Memory is injected here
    ("human", "{input}\n{agent_scratchpad}")
])


# B. Initialize the Agent
agent = initialize_agent(
    llm=llm,
    tools=tools,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    prompt=prompt,
    memory=memory
)

# --- 4. Streamlit Chat UI ---

def run_bot_ui():
    """Streamlit interface for the conversational bot."""
    st.title("🧠 LangChain Conversational Knowledge Bot")
    if "agent" not in st.session_state:
        st.session_state.agent = agent

    if "memory" not in st.session_state:
        st.session_state.memory = memory

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "ai", "content": "Hello! I am a factual knowledge bot. Ask me anything!"}]

    # Display chat messages from history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Handle user input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state["messages"].append({"role": "human", "content": prompt})
        with st.chat_message("human"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.run(prompt)
                except Exception as e:
                    response = f"An unexpected error occurred: {e}"

                st.markdown(response)
                st.session_state["messages"].append({"role": "ai", "content": response})

if __name__ == "__main__":
    run_bot_ui()