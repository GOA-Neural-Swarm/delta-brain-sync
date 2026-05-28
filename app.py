import streamlit as st
import numpy as np
import pandas as pd
from brain import association_rule_mining, NeuralBrain, SovereignArchitect, hyperdimensional_logic_integration, utilitarian_optimization, existential_evolving_process
import os
from PIL import Image
import json
import requests
import time

st.set_page_config(page_title='Delta Brain Sync', page_icon='🐺', layout='wide', initial_sidebar_state='expanded')
st.markdown('\n    <style>\n    .main {\n        background-color: #0e1117;\n        color: #ffffff;\n    }\n    .stSidebar {\n        background-color: #161b22;\n    }\n    h1, h2, h3 {\n        color: #58a6ff;\n    }\n    .stButton>button {\n        width: 100%;\n        background-color: #238636;\n        color: white;\n    }\n    </style>\n    ', unsafe_allow_html=True)

with st.sidebar:
    if os.path.exists('logo.png'):
        logo = Image.open('logo.png')
        st.image(logo, width='stretch')
    st.title('Delta Brain Sync (Agentic)')
    st.markdown('---')
    st.subheader('🔑 Bring Your Own Key')
    api_provider = st.selectbox('Select AI Provider', ['Groq', 'Gemini', 'OpenAI'])
    user_api_key = st.text_input(f'Enter your {api_provider} API Key', type='password')
    if user_api_key:
        st.success(f'{api_provider} API Key Active')
    else:
        st.warning('API Key required to unlock processing power.')
    st.markdown('---')
    st.subheader('💎 Premium Access / Support the Creator')
    st.markdown('\n        To unlock advanced automated analytics, send $10 (USDT) to this TRC-20 Wallet Address: \n        `[YOUR_FUTURE_CRYPTO_ADDRESS_HERE]`\n        ')

st.title('🐺 Delta Brain Sync: Swarm Intelligence Dashboard')
st.write('Welcome to the next generation of association rule mining and Agentic Neural Evolution.')

tab1, tab2, tab3 = st.tabs(['🤖 Agentic Data Miner', '🧠 Neural Evolution', '⚙️ System Health'])

# ----------------- UTILS & CORE TOOLS -----------------
def parse_transactions(data_input):
    try:
        transactions = json.loads(data_input)
    except:
        transactions = [line.split(',') for line in data_input.strip().split('\n') if line]
    return transactions

# 🛠️ Define Tools for Manus-style Agent
MANUS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_association_mining",
            "description": "Perform association rule mining on given transactions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transactions": {"type": "string", "description": "JSON string of transactions (e.g. '[[1, 2], [3, 4]]')"},
                    "min_support": {"type": "integer", "description": "Minimum support value (e.g. 2)"}
                },
                "required": ["transactions", "min_support"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "trigger_hyperdimensional_pca",
            "description": "Run PCA and Neural Core classification on the data to find hyperdimensional logic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "JSON string of numerical data arrays (e.g. '[[1, 2, 3], [4, 5, 6]]')"}
                },
                "required": ["data"]
            }
        }
    }
]

# Tool Execution Mapping
def execute_tool(tool_name, tool_args):
    try:
        if tool_name == "run_association_mining":
            trans = json.loads(tool_args["transactions"])
            min_sup = tool_args["min_support"]
            rules = association_rule_mining(trans, min_sup)
            return json.dumps({"status": "success", "rules": rules})
        
        elif tool_name == "trigger_hyperdimensional_pca":
            data = json.loads(tool_args["data"])
            brain = NeuralBrain()
            result = hyperdimensional_logic_integration(brain, data, return_output=True) # Modified in brain.py to return string
            return json.dumps({"status": "success", "pca_neural_output": result})
            
        else:
            return json.dumps({"error": f"Tool '{tool_name}' not found."})
    except Exception as e:
        return json.dumps({"error": str(e)})

# 🤖 Agentic ReAct Loop (Production Grade)
def run_agentic_loop(prompt, transactions_raw, api_key):
    st.markdown("### 🧠 Agentic Thought Process")
    log_container = st.container()
    
    messages = [
        {"role": "system", "content": "You are an autonomous AI Agent managing the Delta Brain Sync system. You have access to data mining tools and neural classifiers. Analyze the user's request, use the provided tools step-by-step to process the data, and provide a final cyber intelligence report. Data context is provided by the user."},
        {"role": "user", "content": f"User Request: {prompt}\n\nAvailable Data Context: {transactions_raw}"}
    ]
    
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    max_steps = 5
    
    for step in range(max_steps):
        payload = {
            "model": "llama-3.3-70b-versatile", # Using larger model for agentic reasoning
            "messages": messages,
            "tools": MANUS_TOOLS,
            "tool_choice": "auto",
            "temperature": 0.2
        }
        
        try:
            response = requests.post('https://api.groq.com/openai/v1/chat/completions', headers=headers, json=payload).json()
            
            if 'error' in response:
                log_container.error(f"❌ API Error: {response['error']['message']}")
                return None
                
            message = response['choices'][0]['message']
            messages.append(message)
            
            # Action: Tool Calling
            if message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    args_str = tool_call["function"]["arguments"]
                    
                    log_container.info(f"🛠️ **Agent Action:** Calling `{tool_name}` with args: `{args_str}`")
                    
                    # Execute
                    tool_result = execute_tool(tool_name, json.loads(args_str))
                    log_container.success(f"✅ **Tool Output:** `{tool_result}`")
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_name,
                        "content": tool_result
                    })
            else:
                # Observation / Final Answer
                st.markdown("---")
                st.subheader('🛰️ Final Agentic Intelligence Report')
                st.markdown(message["content"])
                return message["content"]
                
        except Exception as e:
            log_container.error(f"❌ Execution Error: {str(e)}")
            return None
            
    log_container.warning("⚠️ Agent reached maximum steps without concluding.")
    return None

def initialize_boot_sequence():
    try:
        architect = SovereignArchitect()
        architect.boot_sequence()
        st.code('--- Sovereign Omni-Sync Architect Initialized ---\nGen Level: 31\nNeural Memory: Syncing...')
        st.success('Architect Ready & Evolved.')
    except Exception as e:
        st.error(f'Evolution Error: {e}')

# ----------------- TABS -----------------
with tab1:
    st.header('Agentic Association & Neural Mining')
    st.write('Instruct the Agent to autonomously mine, classify, and analyze your data.')
    
    data_input = st.text_area('Enter your data (JSON format or comma-separated lists)', placeholder='[[1, 2, 3], [4, 5, 6], [7, 8, 9]]', height=150)
    user_prompt = st.text_input('Agent Instruction', value='Extract association rules with minimum support 2, then run a hyperdimensional PCA analysis on the data to classify the phenomena. Give me a final executive summary.')
    
    if st.button('Execute Agentic Swarm'):
        if not user_api_key:
            st.error('Please enter your Groq API Key in the sidebar first.')
        else:
            transactions = parse_transactions(data_input)
            if not transactions:
                st.error('Please provide valid transaction data.')
            else:
                with st.spinner('Agent is reasoning and executing...'):
                    run_agentic_loop(user_prompt, json.dumps(transactions), user_api_key)

with tab2:
    st.header('Neural Evolution')
    st.write('Monitor and trigger Sovereign Architect evolution cycles.')
    if st.button('Initialize Boot Sequence'):
        if not user_api_key:
            st.error('Please enter your API Key in the sidebar first.')
        else:
            initialize_boot_sequence()

with tab3:
    st.header('System Health')
    col1, col2 = st.columns(2)
    with col1:
        st.metric('Generation Level', '31', '+1') # Updated to Gen 31 based on brain.py
    with col2:
        st.metric('Stability Rating', '100%', 'Secure')
    st.write('### Evolution Logs')
    if os.path.exists('evolution_logs.md'):
        with open('evolution_logs.md', 'r') as f:
            st.markdown(f.read())
    else:
        st.info('No logs found yet. Run an evolution cycle to generate logs.')

st.markdown('---')
st.markdown('2026 Delta Brain Sync | Powered by Sovereign Omni-Sync Architect (Agentic Core)')
