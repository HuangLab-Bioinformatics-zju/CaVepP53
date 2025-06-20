import base64
from Bio.PDB import MMCIFParser, Superimposer
import json
import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage, SystemMessage
from dotenv import load_dotenv
from prompt import system_prompt
from graph_deepseek import invoke_our_graph as invoke_deepseek_graph
from util_deepseek import display_message as display_message_deepseek, render_conversation_history as render_conversation_history_deepseek, get_conversation_summary as get_conversation_summary_deepseek
from datetime import datetime
from tools import compare_structure


# Load environment variables
load_dotenv()

# Initialize session state if not present
if "page" not in st.session_state:
    st.session_state["page"] = "OpenAI"

if "final_state" not in st.session_state:
    st.session_state["final_state"] = {
        "messages": [SystemMessage(content=system_prompt)]
    }

# Add custom CSS with theme-aware styling
st.markdown("""
<style>
    /* Custom styling for the main title */
    .main-title {
        text-align: center;
        color: #89BCE4;
        padding: 1rem 0;
        border-bottom: 2px solid #89BCE4;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    /* Provider selection styling */
    .provider-section {
        background-color: var(--secondary-background-color);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 25px;
        background-color: #89BCE4;
        color: white;
        font-weight: 600;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    .new-chat-button > button {
        background-color: #2196F3 !important;
        margin: 1rem 0;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 8px 16px;
    }
    .stTabs [data-baseweb="tab"][data-selected] {
        background-color: #E5B2BD; 
        color: #FFFFFF; 
        border-color: #E5637B;
        border-bottom: 1px solid #E5637B; 
    }    
    /* Chat message styling */
    .chat-message {
        padding: 1.75rem;
        border-radius: 12px;
        margin: 1.25rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .user-message {
        background-color: #FBE9E7;
        border-left: 4px solid #FF5722;
    }
    
    .ai-message {
        background-color: #E8F5E9;
        border-left: 4px solid #2196F3;
    }
    
    /* Form styling */
    .stForm {
        background-color: var(--background-color);
        padding: 1.75rem;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
    }  
    /* Image upload area styling */
    [data-testid="stFileUploader"] {
        background-color: var(--background-color);
        padding: 1.25rem;
        border-radius: 12px;
        border: 2px dashed #89BCE4;
    }
    
    /* Dark mode styles */
    @media (prefers-color-scheme: dark) {
        .main-title {
            color: #FFAB91;
            border-bottom-color: #89BCE4;
        }
        
        .provider-section {
            background-color: #1E1E1E;
        }
        
        .user-message {
            background-color: #3E2723;
            border-left: 4px solid #FFAB91;
        }
        
        .ai-message {
            background-color: #1A237E;
            border-left: 4px solid #90CAF9;
        }
    }
    
    /* Input field styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        padding: 12px 24px;
        border: 2px solid #89BCE4;
        font-size: 16px;
    }
    
    /* Submit button hover effect */
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
        background-color: #89BCE4;
        color: #E5637B; 
        border-color: #E5637B;
    }

    /* Tab hover effect */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #89BCE4;
        transition: background-color 0.3s ease;
    }

    /* API key setup styling */
    .api-key-setup {
        background-color: var(--secondary-background-color);
        padding: 1.25rem;
        border-radius: 12px;
        margin: 1.25rem 0;
        border: 1px solid #89BCE4;
    }
    /* Main chat interface title styling */
    .chat-title {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 18px;
        background: linear-gradient(90deg, #89BCE4, transparent);
        border-radius: 12px;
        margin-bottom: 24px;
    }
    
    .robot-icon {
        font-size: 28px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.2); }
        100% { transform: scale(1); }
    }
    
    .provider-name {
        color: #FF5722;
        font-weight: bold;
        font-size: 18px;
    }
</style>
""", unsafe_allow_html=True)

# Set up Streamlit layout
st.markdown('<h1 class="main-title">🤖 TP53 VEP Interpreting AI Agent</h1>', unsafe_allow_html=True)
import streamlit as st

# 定义工具列表
tools = [
    "Search_PubMed",
    "Get_UniProt_Info",
    "Get_TP53_Info",
    "Get_MSA_Info",
    "Get_Virtual_Perturbation_Info",
    "Compare_Structure",
    "Report_Tool"
]

st.markdown("""
<style>
    .provider-section {
        background-color: var(--secondary-background-color);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    .tools-list {
        list-style-type: none;
        padding: 0;
    }
    .tools-list li {
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(0,0,0,0.1);
    }
    .tools-list li:last-child {
        border-bottom: none;
    }
</style>
""", unsafe_allow_html=True)

#st.sidebar.markdown('<div class="provider-section">', unsafe_allow_html=True)
st.sidebar.title('🛠️ Tools List')
st.sidebar.markdown('<ul class="tools-list">', unsafe_allow_html=True)

for tool in tools:
    st.sidebar.markdown(f'<li>{tool}</li>', unsafe_allow_html=True)

st.sidebar.markdown('</ul>', unsafe_allow_html=True)
st.sidebar.markdown('<p style="text-align: center; color: #5979A2; font-size: 14px;">' +
                    'You can call upon these tools within the conversation box.</p>', unsafe_allow_html=True)


# Update the provider selection
st.sidebar.title("🗳️ Select Deepseek Model")
#selected = st.sidebar.radio("Provider", provider_options)
page = 'DeepSeek'
st.session_state["page"] = 'DeepSeek'


HISTORY_DIR = "conversation_histories_deepseek"
invoke_graph = invoke_deepseek_graph
display_message = display_message_deepseek
render_conversation_history = render_conversation_history_deepseek
get_conversation_summary = get_conversation_summary_deepseek
available_models = [
    "deepseek-chat", 
    "deepseek-reasoner"
]

# Add model selection with improved styling
selected_model = st.sidebar.selectbox('√ Select', available_models, index=0)


# Add New Chat button with custom styling
st.sidebar.markdown('<div class="new-chat-button">', unsafe_allow_html=True)
if st.sidebar.button("🔄 Start New Chat"):
    st.session_state["final_state"] = {
        "messages": [SystemMessage(content=system_prompt)]
    }
    st.session_state["last_summary_point"] = 0
    st.session_state["last_summary_title"] = "Default Title"
    st.session_state["last_summary_summary"] = "This is the default summary for short conversations."
    st.rerun()
st.sidebar.markdown('</div>', unsafe_allow_html=True)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Set up environment for API keys
if page == "DeepSeek" and not os.getenv('DEEPSEEK_API_KEY'):
    st.sidebar.header("DeepSeek API Key Setup")
    api_key = st.sidebar.text_input(label="DeepSeek API Key", type="password", label_visibility="collapsed")
    os.environ["DEEPSEEK_API_KEY"] = api_key
    if not api_key:
        st.info("Please enter your DeepSeek API Key in the sidebar.")
        st.stop()

os.makedirs(HISTORY_DIR, exist_ok=True)


def save_history(title: str, summary: str):
    """Save the current conversation history to a file with title and summary."""
    history_data = {
        "title": title,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
        "messages": messages_to_dicts(st.session_state["final_state"]["messages"])
    }
    filename = f"{HISTORY_DIR}/{title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(history_data, f)
    st.rerun()

def load_all_histories():
    """Load all saved conversation histories as a list of metadata for display."""
    histories = []
    for file in os.listdir(HISTORY_DIR):
        if file.endswith(".json"):
            with open(os.path.join(HISTORY_DIR, file), "r") as f:
                history = json.load(f)
                histories.append({
                    "title": history["title"],
                    "summary": history["summary"],
                    "timestamp": history["timestamp"],
                    "filename": file
                })
    return sorted(histories, key=lambda x: x["timestamp"], reverse=True)

def load_history(filename: str):
    """Load a specific conversation history file into session state."""
    try:
        with open(os.path.join(HISTORY_DIR, filename), "r") as f:
            history_data = json.load(f)
            st.session_state["final_state"]["messages"] = dicts_to_messages(history_data["messages"])
        st.sidebar.success(f"Conversation '{history_data['title']}' loaded successfully")
    except FileNotFoundError:
        st.sidebar.error("Conversation history not found.")

def delete_history(filename: str):
    """Delete a specific conversation history file."""
    os.remove(os.path.join(HISTORY_DIR, filename))
    st.sidebar.success("Conversation history deleted.")
    st.rerun()

# Convert messages to serializable dictionaries and vice versa
def messages_to_dicts(messages):
    return [msg.dict() for msg in messages]

def dicts_to_messages(dicts):
    reconstructed_messages = []
    for d in dicts:
        if d["type"] == "ai":
            reconstructed_messages.append(AIMessage(**d))
        elif d["type"] == "human":
            reconstructed_messages.append(HumanMessage(**d))
        elif d["type"] == "tool":
            reconstructed_messages.append(ToolMessage(**d))
    return reconstructed_messages
# Organize Sidebar with Tabs and improved styling
st.sidebar.title("↕️ Upload and Download")
tab1, tab2 = st.sidebar.tabs(["🗂️ CIF File Upload", "💬 Conversation Management"])
# Initialize session state variables
if "last_summary_point" not in st.session_state:
    st.session_state["last_summary_point"] = 0
if "last_summary_title" not in st.session_state:
    st.session_state["last_summary_title"] = "Default Title"
if "last_summary_summary" not in st.session_state:
    st.session_state["last_summary_summary"] = "This is the default summary for short conversations."


# Tab 1: CIF File Upload
with tab1:
    st.subheader("Protein Structure File")
    with st.form("file_upload_form", clear_on_submit=True):
        uploaded_file = st.file_uploader("Upload the mutation CIF file", type=["cif"], accept_multiple_files=False)
        submitted = st.form_submit_button("Submit Files")
        if submitted:
            if uploaded_file:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                tmp_dir = 'tmp'
                os.makedirs(tmp_dir, exist_ok=True)
                file_path = os.path.join(tmp_dir, uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                st.session_state["uploaded_file"] = [
                    file_path
                ]
                st.success("CIF files uploaded successfully!")
            else:
                st.session_state["uploaded_file"] = []
                st.warning("No CIF files uploaded.")


# Tab 2: Conversation Management
with tab2:
    st.subheader("History")
    histories = load_all_histories()
    if histories:
        st.markdown("### Saved Histories")
        for history in histories:
            with st.expander(f"{history['title']} ({history['timestamp'][:10]})"):
                st.write(history["summary"])
                if st.button("Load", key=f"load_{history['filename']}"):
                    load_history(history["filename"])
                if st.button("Delete", key=f"delete_{history['filename']}"):
                    delete_history(history["filename"])

    # Determine title and summary based on message count and last summary point
    message_count = len(st.session_state["final_state"]["messages"])
    if message_count > 5 and (message_count - 5) % 10 == 0 and message_count != st.session_state["last_summary_point"]:
        st.session_state["last_summary_title"] = "Default Title"
        st.session_state["last_summary_summary"] = "This is the default summary for short conversations."
        st.session_state["last_summary_point"] = message_count
    elif message_count <= 5:
        st.session_state["last_summary_title"] = "Default Title"
        st.session_state["last_summary_summary"] = "This is the default summary for short conversations."

    title = st.text_input("Conversation Title", value=st.session_state["last_summary_title"])
    summary = st.text_area("Conversation Summary", value=st.session_state["last_summary_summary"])

    if st.button("Save Conversation"):
        save_history(title, summary)
        st.sidebar.success(f"Conversation saved as '{title}'")



# Main chat interface
st.markdown(f"""
    <div class="chat-title">
        <span class="robot-icon">🤖</span>
        <span>Chat with CaVepP53 Agent</span>
    </div>
""", unsafe_allow_html=True)

if "final_state" in st.session_state and "messages" in st.session_state["final_state"]:
    with st.container():
        render_conversation_history(st.session_state["final_state"]["messages"])

# Initialize prompt variable
prompt = st.session_state.get("audio_transcription")

render_conversation_history(st.session_state["final_state"]["messages"][0:])

# Capture text input if no audio input
if prompt is None:
    prompt = st.chat_input()

# Process new user input if available
if prompt:
    content_list = [{"type": "text", "text": prompt}]

    user_message = HumanMessage(content=content_list)
    st.session_state["final_state"]["messages"].append(user_message)
    render_conversation_history([user_message])

    with st.spinner(f"Agent is thinking..."):
        previous_message_count = len(st.session_state["final_state"]["messages"])
        updated_state = invoke_graph(st.session_state["final_state"]["messages"], selected_model)
    
    st.session_state["final_state"] = updated_state
    new_messages = st.session_state["final_state"]["messages"][previous_message_count:]
    
    if st.session_state.get("render_last_message", True):
        render_conversation_history([st.session_state["final_state"]["messages"][-1]])
    
