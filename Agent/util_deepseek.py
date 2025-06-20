import os
import json
import time
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_deepseek import ChatDeepSeek  
from typing import List, Tuple
from pydantic import BaseModel, Field
# Directory for temporary plot images
plot_dir = os.path.join(os.path.dirname(__file__), "tmp/plots")
os.makedirs(plot_dir, exist_ok=True)

def display_message(content, sender="assistant"):
    """
    Displays a message from the user or assistant with different styling.
    Supports displaying both text and image URLs for the user.
    """
    if sender == "user":
        if isinstance(content, str):
            # Display plain text message from user
            st.markdown(
                f"""
                <div style="text-align: right;">
                    <div style="display: inline-block; background-color: #DCF8C6; color: #000; padding: 10px; border-radius: 15px; margin: 5px; max-width: 60%; text-align: left;">
                        <p style="margin: 0;">{content}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif isinstance(content, dict):
            # Check if the content has both text and image URL
            if "text" in content:
                st.markdown(
                    f"""
                    <div style="text-align: right;">
                        <div style="display: inline-block; background-color: #DCF8C6; color: #000; padding: 10px; border-radius: 15px; margin: 5px; max-width: 60%; text-align: left;">
                            <p style="margin: 0;">{content["text"]}</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            if "url" in content:
                st.image(content["url"], caption="User Image", use_container_width=True)
    else:
        # Display assistant's message, handling various content formats
        if isinstance(content, str):
            # Process and display plain text with LaTeX-style content
            modified_content = content.replace("\\(", "$").replace("\\)", "$")
            modified_content = modified_content.replace("\\[", "$$").replace("\\]", "$$")
            st.markdown(modified_content)
        elif isinstance(content, list):
            # Iterate through list items if content is a list
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text" and "text" in item:
                        # Process LaTeX-style text
                        modified_text = item["text"].replace("\\(", "$").replace("\\)", "$")
                        modified_text = modified_text.replace("\\[", "$$").replace("\\]", "$$")
                        st.markdown(modified_text)
                    elif "url" in item:
                        st.image(item["url"], caption="Assistant Image", use_container_width=True)
                elif isinstance(item, str):
                    # Handle plain text items in the list
                    modified_content = item.replace("\\(", "$").replace("\\)", "$")
                    modified_content = modified_content.replace("\\[", "$$").replace("\\]", "$$")
                    st.markdown(modified_content)
        elif isinstance(content, dict):
            # Display text if present in a single dictionary
            if "text" in content:
                modified_text = content["text"].replace("\\(", "$").replace("\\)", "$")
                modified_text = modified_text.replace("\\[", "$$").replace("\\]", "$$")
                st.markdown(modified_text)
            if "url" in content:
                st.image(content["url"], caption="Assistant Image", use_container_width=True)
        else:
            # Handle unexpected content type
            st.error("Unsupported content format from the assistant.")


def render_conversation_history(messages):
    """
    Renders conversation history from a list of messages, handling multiple tool calls.
    """
    tool_input_map = {}  # Map to track tool_call_id to tool_input
    
    for entry in messages:
        # Skip if the message has name "image_assistant"
        if hasattr(entry, "name") and entry.name == "image_assistant":
            continue

        if isinstance(entry, HumanMessage):
            # Check if entry.content is list or string and handle appropriately
            if isinstance(entry.content, list):
                for item in entry.content:
                    if isinstance(item, dict):
                        # Display text or image URL in dictionary format
                        if item["type"] == "text":
                            display_message(item["text"], sender="user")
                        elif item["type"] == "image_url":
                            display_message({"url": item["image_url"]["url"]}, sender="user")
                    elif isinstance(item, str):
                        # Display plain text if it's a string
                        display_message(item, sender="user")
            elif isinstance(entry.content, str):
                # Display single string content
                display_message(entry.content, sender="user")

        elif isinstance(entry, AIMessage):
            display_message(entry.content, sender="assistant")
            
            # Handle tool calls in AIMessage
            if entry.tool_calls:
                tool_calls = entry.tool_calls
                for tool_call in tool_calls:
                    try:
                        arguments_json = tool_call.get('args', '{}')
                        #tool_input = arguments_json.get("query", "No tool input available")
                        tool_input = arguments_json 
                        tool_call_id = tool_call.get("id")
                        if tool_call_id:
                            tool_input_map[tool_call_id] = tool_input
                    except json.JSONDecodeError:
                        tool_input_map[tool_call.get("id", "unknown")] = "Error decoding tool input."

        elif isinstance(entry, ToolMessage):
            display_tool_message(entry, tool_input_map)


def display_tool_message(entry, tool_input_map):
    """
    Display a tool message with the corresponding tool input based on the tool_call_id.
    """
    tool_output = entry.content
    tool_call_id = getattr(entry, "tool_call_id", None)  # Get the tool_call_id from the ToolMessage
    tool_input = tool_input_map.get(tool_call_id, "No matching tool input found")  # Match with tool_input_map

    with st.expander(f"Tool Call: {entry.name}", expanded=False):
        # Check if 'query' exists in tool_input and display accordingly
        if isinstance(tool_input, dict) and 'query' in tool_input:
            st.code(tool_input['query'], language="python")
        else:
            st.code(tool_input or "No tool input available", language="python")
        st.write("**Tool Output:**")
        st.code(tool_output)
        
        # Handle artifacts if they exist
        artifacts = getattr(entry, "artifact", [])
        if artifacts:
            st.write("**Generated Artifacts (e.g., Plots):**")
            for rel_path in artifacts:
                if rel_path.endswith(".png"):
                    # Convert relative path to absolute
                    abs_path = os.path.join(os.path.dirname(__file__), rel_path)
                    if os.path.exists(abs_path):
                        st.image(abs_path, caption="Generated Plot")
                    else:
                        st.write(f"Error: Plot file not found at {rel_path}")


# Pydantic model for structured output
class ConversationSummary(BaseModel):
    """Structure for conversation title and summary."""
    title: str = Field(description="The title of the conversation")
    summary: str = Field(description="A concise summary of the conversation's main points")

# Function to get conversation title and summary
def get_conversation_summary(messages: List[BaseMessage]) -> Tuple[str, str]:
    # Initialize the LLM model within the function
    llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

    # Define the prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("msgs"),
        ("human", "Given the above messages between user and AI agent, return a title and concise summary of the conversation"),
    ])

    # Configure the structured output model
    structured_llm = llm.with_structured_output(ConversationSummary)
    summarized_chain = prompt_template | structured_llm

    # Invoke the chain with the messages and retrieve the response
    response = summarized_chain.invoke(messages)
    
    # Return the title and summary
    return response.title, response.summary