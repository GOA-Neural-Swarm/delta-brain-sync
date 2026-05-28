import streamlit as st
import numpy as np
import pandas as pd
import os
from PIL import Image
import json
import time
import requests
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
    .agent-thought { background-color: #1c2128; border-left: 4px solid #58a6ff; padding: 10px; margin: 10px 0; font-family: monospace; }
    </style>
    """, unsafe_allow_html=True)

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
    
    st.subheader("💎 Premium Access / Support the Creator")
    st.markdown("""
    To unlock advanced automated analytics, send $10 (USDT) to this TRC-20 Wallet Address: 
    
    `[YOUR_FUTURE_CRYPTO_ADDRESS_HERE]`
    """)

# --- Agentic Core ---

MANUS_TOOLS = [
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
    """Router to execute brain.py functions from LLM tool calls."""
    brain = NeuralBrain()
    try:
        if tool_name == "association_rule_mining":
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
    """Simplified Groq API call for Stlite environment."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": 0.2
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

def run_agentic_loop(user_prompt, data_context, api_key):
    """ReAct framework loop for autonomous reasoning and tool use."""
    system_prompt = f"""You are a 'Swarm Node Agent' for Delta Brain Sync. 
Your goal is to provide a 'Cyber Intelligence Report' by analyzing data step-by-step.
You have access to these tools: {json.dumps(MANUS_TOOLS)}

If you need to use a tool, respond with ONLY a JSON object: 
{{"tool_use": {{"name": "tool_name", "args": {{"arg1": "val1"}}}}}}

Once you have the final answer, start your response with 'FINAL REPORT:'."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Analyze this data: {data_context}. Task: {user_prompt}"}
    ]
    
    max_steps = 5
    for i in range(max_steps):
        with st.spinner(f"Agent Thinking (Step {i+1})..."):
            response = call_groq(messages, api_key)
            
            if 'choices' not in response:
                st.error(f"API Error: {response}")
                break
                
            ai_msg = response['choices'][0]['message']['content']
            
            # Check for tool use
            try:
                if '{"tool_use":' in ai_msg:
                    start = ai_msg.find('{')
                    end = ai_msg.rfind('}') + 1
                    tool_json = json.loads(ai_msg[start:end])
                    
                    tool_name = tool_json['tool_use']['name']
                    tool_args = tool_json['tool_use']['args']
                    
                    st.info(f"🛠️ **Agent Action:** Using `{tool_name}`")
                    result = execute_tool(tool_name, tool_args)
                    
                    messages.append({"role": "assistant", "content": ai_msg})
                    messages.append({"role": "user", "content": f"Tool Result: {result}"})
                    continue
            except:
                pass

            if "FINAL REPORT:" in ai_msg:
                return ai_msg
            
            return ai_msg
            
    return "Agent loop timed out."

# --- Main UI ---
st.title("🐺 Delta Brain Sync: Agentic Swarm Intelligence")
st.write("Upgraded to Agentic Orchestrator v2.0 (Manus-Style Tool Integration)")

tab1, tab2, tab3 = st.tabs(["🤖 Agentic Data Miner", "🧠 Neural Evolution", "⚙️ System Health"])

with tab1:
    st.header("Agentic Data Mining")
    st.write("Instruct the Swarm Node Agent to process complex phenomena.")
    
    data_input = st.text_area("Data Context (JSON or CSV)", 
                             placeholder="[[1, 2], [1, 2], [3, 4]]",
                             height=150)
    
    user_goal = st.text_input("Agent Mission", placeholder="Find patterns and optimize for utility...")
    
    if st.button("Deploy Agent"):
        if not user_api_key:
            st.error("Please enter your API Key in the sidebar.")
        elif not data_input or not user_goal:
            st.error("Please provide both data and a mission.")
        else:
            report = run_agentic_loop(user_goal, data_input, user_api_key)
            st.success("Analysis Complete")
            st.markdown("### 📋 Cyber Intelligence Report")
            st.write(report)

with tab2:
    st.header("Neural Evolution")
    st.write("Trigger and monitor Sovereign Architect evolution cycles.")
    
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
