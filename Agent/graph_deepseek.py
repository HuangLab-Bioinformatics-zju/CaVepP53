import os
import base64
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Annotated, TypedDict, Literal, Tuple, List
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
#from langchain_experimental.utilities import PythonREPL
from tools import PythonREPL
from langgraph.prebuilt import ToolNode
from prompt import system_prompt
from pydantic import BaseModel, Field
from langgraph.types import Command
from textwrap import dedent
import streamlit as st
from util_deepseek import display_message, render_conversation_history, get_conversation_summary
from langchain_core.runnables.config import RunnableConfig
from tools import pubmed_rag_agent, get_uniprot_info, get_tp53_info, get_MSA_info, get_virtual_perturbation_info, compare_structure, report_tool
import sys
import io
load_dotenv()

python_repl = PythonREPL()

@tool(response_format="content_and_artifact")
def python_repl_tool(query: str) -> Tuple[str, List[str]]:
    """A Python shell. Use this to execute python commands. Input should be a valid python command. 
    If you want to see the output of a value, you should print it out with `print(...)`. """
    
    plot_paths = []  # List to store file paths of generated plots
    result_parts = []  # List to store different parts of the output
    
    try:
        output = python_repl.run(query)
        if output and output.strip():
            result_parts.append(output.strip())
        
        figures = [plt.figure(i) for i in plt.get_fignums()]
        if figures:
            for fig in figures:
                fig.set_size_inches(10, 6)  # Ensure figures are large enough
                #fig.tight_layout()  # Prevent truncation# Generate filename
                plot_filename = f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
                # Create relative path
                rel_path = os.path.join("tmp/plots", plot_filename)
                # Convert to absolute path for saving
                abs_path = os.path.join(os.path.dirname(__file__), rel_path)
                
                fig.savefig(abs_path,bbox_inches='tight')
                plot_paths.append(rel_path)  # Store relative path
            
            plt.close("all")
            result_parts.append(f"Generated {len(plot_paths)} plot(s).")
        
        if not result_parts:  # If no output and no figures
            result_parts.append("Executed code successfully with no output. If you want to see the output of a value, you should print it out with `print(...)`.")

    except Exception as e:
        result_parts.append(f"Error executing code: {e}")
    
    # Join all parts of the result with newlines
    result_summary = "\n".join(result_parts)
    
    # Return both the summary and plot paths (if any)
    return result_summary, plot_paths
# Tools List and Node Setup
tools = [
    pubmed_rag_agent,
    get_uniprot_info,
    get_tp53_info, 
    get_MSA_info, 
    get_virtual_perturbation_info, 
    compare_structure,
    report_tool
]
tool_node = ToolNode(tools)

# Graph Setup
class GraphsState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    input_messages_len: list[int]

graph = StateGraph(GraphsState)

deepseek_chat = ChatDeepSeek(model_name="deepseek-chat",temperature=0,max_tokens=8000).bind_tools(tools)
deepseek_reasoner = ChatDeepSeek(model_name="deepseek-reasoner",temperature=0,max_tokens=8000).bind_tools(tools)


models = {
    "deepseek-chat": deepseek_chat,
    "deepseek-reasoner": deepseek_reasoner
}

def _call_model(state: GraphsState, config: RunnableConfig) -> Command[Literal["tools", "__end__"]]:
    st.session_state["final_state"]["messages"]=state["messages"]
    model_name = config["configurable"].get("model", "deepseek_chat")
    llm = models[model_name]
    previous_message_count = len(state["messages"])
    state["input_messages_len"].append(previous_message_count)
    render_conversation_history(state["messages"][state["input_messages_len"][-2]:state["input_messages_len"][-1]])
    cur_messages_len = len(state["messages"])-state["input_messages_len"][0]  
    if cur_messages_len > 200:
        st.markdown(
        f"""
        <p style="color:blue; font-size:16px;">
            Current recursion step is {cur_messages_len}. Terminated because you exceeded the limit of 200.
        </p>
        """,
        unsafe_allow_html=True
        )
        st.session_state["render_last_message"] = False
        return Command(
        update={"messages": []},
        goto="__end__",
    )


    if "uploaded_file" in st.session_state and st.session_state["uploaded_file"]:
        cif_data = st.session_state["uploaded_file"][0]
        prompt_msg = HumanMessage(content=f"""
        Data:
        {cif_data}
        """)
        state["messages"].append(prompt_msg)
        st.session_state["uploaded_file"] = []

    last_message = state["messages"][-1]
    response = llm.invoke(state["messages"])
    if response.tool_calls:
        return Command(
        update={"messages": [response]},
        goto="tools",
    )
    else:
        st.session_state["render_last_message"] = True
        return Command(
        update={"messages": [response]},
        goto="__end__",
    )

graph.add_edge(START, "modelNode")
graph.add_node("tools", tool_node)
graph.add_node("modelNode", _call_model)
graph.add_edge("tools", "modelNode")
graph_runnable = graph.compile()

def invoke_our_graph(messages,model_choose):
    config = {"recursion_limit": 200, "configurable": {"model": model_choose}}
    return graph_runnable.invoke({"messages": messages,"input_messages_len":[len(messages)]},config=config)