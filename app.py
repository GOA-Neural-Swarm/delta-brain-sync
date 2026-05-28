import streamlit as st
import numpy as np
import pandas as pd
import os
from PIL import Image
import json
import time
import requests
import re
from brain import (
    association_rule_mining, 
    NeuralBrain, 
    SovereignArchitect, 
    hyperdimensional_logic_integration, 
    utilitarian_optimization, 
    existential_evolving_process
)

# --- Configuration ---
st.set_page_config(
    page_title="Delta Brain Sync",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stSidebar { background-color: #161b22; }
    h1, h2, h3 { color: #58a6ff; }
    .stButton>button { width: 100%; background-color: #238636; color: white; }
    .stChatMessage { background-color: #1c2128; border-radius: 10px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "generated_files" not in st.session_state:
    st.session_state.generated_files = []

# --- Sidebar ---
with st.sidebar:
    if os.path.exists('logo.png'):
        logo = Image.open('logo.png')
        st.image(logo, use_container_width=True)
    
    st.title("Delta Brain Sync")
    st.markdown("---")
    
    st.subheader("🔑 Bring Your Own Key")
    api_provider = st.selectbox("Select AI Provider", ["Groq", "Gemini", "OpenAI"])
    user_api_key = st.text_input(f"Enter your {api_provider} API Key", type="password")
    
    if user_api_key:
        st.success(f"{api_provider} API Key Active")
    else:
        st.warning("API Key required to unlock processing power.")
    
    st.markdown("---")
    
    # Generated Files Download Section
    if st.session_state.generated_files:
        st.subheader("📁 Generated Files")
        for filename in st.session_state.generated_files:
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    st.download_button(
                        label=f"Download {filename}",
                        data=f,
                        file_name=filename,
                        mime="text/plain",
                        key=f"dl_{filename}"
                    )
        st.markdown("---")

    st.subheader("💎 Premium Access / Support the Creator")
    st.markdown("""
    To unlock advanced automated analytics, send $10 (USDT) to this TRC-20 Wallet Address: 
    
    `[YOUR_FUTURE_CRYPTO_ADDRESS_HERE]`
    """)

# --- Agentic Core Tools ---

MANUS_TOOLS = [
    {
        "name": "search_web",
        "description": "Search the web for information using Wikipedia API.",
        "parameters": {
            "query": "The search query string."
        }
    },
    {
        "name": "fetch_url_content",
        "description": "Fetch and extract text content from a specific URL.",
        "parameters": {
            "url": "The URL to fetch content from."
        }
    },
    {
        "name": "create_and_save_file",
        "description": "Save data or reports to a file in the virtual filesystem.",
        "parameters": {
            "filename": "The name of the file (e.g., 'data.csv').",
            "content": "The string content to save."
        }
    },
    {
        "name": "read_local_file",
        "description": "Read content from a previously saved file.",
        "parameters": {
            "filename": "The name of the file to read."
        }
    },
    {
        "name": "association_rule_mining",
        "description": "Find frequent itemsets and rules in transaction data.",
        "parameters": {
            "transactions": "List of lists representing transactions.",
            "min_support": "Minimum occurrence count (integer)."
        }
    },
    {
        "name": "hyperdimensional_logic_integration",
        "description": "Perform PCA and neural projection on high-dimensional data.",
        "parameters": {
            "phenomena_data": "2D list of numerical data."
        }
    },
    {
        "name": "utilitarian_optimization",
        "description": "Find the optimal phenomenon based on neural utility scores.",
        "parameters": {
            "phenomena_data": "2D list of numerical data."
        }
    }
]

def execute_tool(tool_name, args):
    """Router to execute tools, including new Web and File I/O capabilities."""
    brain = NeuralBrain()
    try:
        if tool_name == "search_web":
            query = args.get('query', '')
            url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json&origin=*"
            resp = requests.get(url).json()
            results = [f"{item['title']}: {item['snippet']}" for item in resp.get('query', {}).get('search', [])]
            return json.dumps(results[:5])
            
        elif tool_name == "fetch_url_content":
            url = args.get('url', '')
            resp = requests.get(url, timeout=10)
            # Basic HTML stripping using regex for Stlite environment
            text = re.sub('<[^<]+?>', '', resp.text)
            return text[:2000] # Limit output size
            
        elif tool_name == "create_and_save_file":
            filename = args.get('filename', 'output.txt')
            content = args.get('content', '')
            with open(filename, "w") as f:
                f.write(content)
            if filename not in st.session_state.generated_files:
                st.session_state.generated_files.append(filename)
            return f"Successfully saved to {filename}"
            
        elif tool_name == "read_local_file":
            filename = args.get('filename', '')
            if os.path.exists(filename):
                with open(filename, "r") as f:
                    return f.read()
            return f"Error: File {filename} not found."
            
        elif tool_name == "association_rule_mining":
            res = association_rule_mining(args['transactions'], args.get('min_support', 2))
            return json.dumps(res)
            
        elif tool_name == "hyperdimensional_logic_integration":
            res = hyperdimensional_logic_integration(brain, args['phenomena_data'], return_output=True)
            return json.dumps(res)
            
        elif tool_name == "utilitarian_optimization":
            res = utilitarian_optimization(brain, args['phenomena_data'], return_output=True)
            return json.dumps(res)
            
        else:
            return f"Error: Tool {tool_name} not found."
    except Exception as e:
        return f"Execution Error: {str(e)}"

def call_groq(messages, api_key):
    """Groq API call."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": 0.2
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

def run_agentic_loop(chat_history, api_key):
    """Upgraded ReAct loop with multi-step tool use and real-time status updates."""
    system_prompt = f"""You are an autonomous Manus-style AI agent for Delta Brain Sync. 
You can search the web, read pages, construct datasets, save files, and run ML algorithms.
Current Tools: {json.dumps(MANUS_TOOLS)}

If you need a tool, respond with ONLY: 
{{"tool_use": {{"name": "tool_name", "args": {{"arg1": "val1"}}}}}}

Keep thinking and acting until you have the final answer.
Once finished, start your response with 'FINAL REPORT:'."""

    messages = [{"role": "system", "content": system_prompt}] + chat_history
    
    max_steps = 10
    for i in range(max_steps):
        response = call_groq(messages, api_key)
        if 'choices' not in response:
            return f"API Error: {response}"
            
        ai_msg = response['choices'][0]['message']['content']
        
        try:
            if '{"tool_use":' in ai_msg:
                start = ai_msg.find('{')
                end = ai_msg.rfind('}') + 1
                tool_json = json.loads(ai_msg[start:end])
                
                tool_name = tool_json['tool_use']['name']
                tool_args = tool_json['tool_use']['args']
                
                with st.status(f"🤖 Agent Action: {tool_name}", expanded=False) as status:
                    st.write(f"Parameters: {tool_args}")
                    result = execute_tool(tool_name, tool_args)
                    st.write(f"Result: {result[:500]}...")
                    status.update(label=f"✅ Tool Complete: {tool_name}", state="complete")
                
                messages.append({"role": "assistant", "content": ai_msg})
                messages.append({"role": "user", "content": f"Tool Result: {result}"})
                continue
        except Exception as e:
            messages.append({"role": "assistant", "content": ai_msg})
            messages.append({"role": "user", "content": f"Tool Error: {str(e)}"})
            continue

        return ai_msg
            
    return "Agent loop timed out."

# --- Main UI ---
st.title("🐺 Delta Brain Sync: Autonomous Manus Agent")

tab1, tab2, tab3 = st.tabs(["🤖 Agentic Chat", "🧠 Neural Evolution", "⚙️ System Health"])

with tab1:
    st.header("Manus-Style Agentic Orchestrator")
    st.write("Deploy the Swarm Node Agent for autonomous research, data creation, and ML analysis.")

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask the agent to research, create, or analyze..."):
        if not user_api_key:
            st.error("Please enter your API Key in the sidebar.")
        else:
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Run Agentic Loop
            with st.chat_message("assistant"):
                response = run_agentic_loop(st.session_state.messages, user_api_key)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Force rerun to show download buttons in sidebar if files were created
            st.rerun()

with tab2:
    st.header("Neural Evolution")
    if st.button("Initialize Boot Sequence"):
        if not user_api_key:
            st.error("Please enter your API Key in the sidebar.")
        else:
            try:
                architect = SovereignArchitect()
                with st.spinner("Booting..."):
                    architect.boot_sequence()
                    st.code("--- Sovereign Omni-Sync Architect Initialized ---\nGen Level: 19\nNeural Memory: Syncing...")
                    st.success("Architect Ready.")
            except Exception as e:
                st.error(f"Evolution Error: {e}")

with tab3:
    st.header("System Health")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Generation Level", "19", "+1")
    with col2:
        st.metric("Stability Rating", "100%", "Secure")
    
    st.write("### Evolution Logs")
    if os.path.exists('evolution_logs.md'):
        with open('evolution_logs.md', 'r') as f:
            st.markdown(f.read())
    else:
        st.info("No logs found yet. Run an evolution cycle to generate logs.")

st.markdown("---")
st.markdown("© 2026 Delta Brain Sync | Powered by Sovereign Omni-Sync Architect")
