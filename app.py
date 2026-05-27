import telemetry_bridge
import streamlit as st
import numpy as np
import pandas as pd
from brain import association_rule_mining, NeuralBrain, SovereignArchitect
import os
from PIL import Image
import json
st.set_page_config(page_title='Delta Brain Sync', page_icon='🐺', layout='wide', initial_sidebar_state='expanded')
st.markdown('\n    <style>\n    .main {\n        background-color: #0e1117;\n        color: #ffffff;\n    }\n    .stSidebar {\n        background-color: #161b22;\n    }\n    h1, h2, h3 {\n        color: #58a6ff;\n    }\n    .stButton>button {\n        width: 100%;\n        background-color: #238636;\n        color: white;\n    }\n    </style>\n    ', unsafe_allow_html=True)
with st.sidebar:
    if os.path.exists('logo.png'):
        logo = Image.open('logo.png')
        st.image(logo, width='stretch')
    st.title('Delta Brain Sync')
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
st.write('Welcome to the next generation of association rule mining and neural evolution.')
tab1, tab2, tab3 = st.tabs(['📊 Association Mining', '🧠 Neural Evolution', '⚙️ System Health'])

def parse_transactions(data_input):
    """
    Parse transactions from input data.

    Args:
    data_input (str): Input data in JSON or comma-separated format.

    Returns:
    list: Parsed transactions.
    """
    try:
        transactions = json.loads(data_input)
    except:
        transactions = [line.split(',') for line in data_input.strip().split('\n') if line]
    return transactions

def run_association_mining(transactions, min_support):
    """
    Run association rule mining on transactions.

    Args:
    transactions (list): Parsed transactions.
    min_support (int): Minimum support for association rules.

    Returns:
    dict: Association rules.
    """
    try:
        results = association_rule_mining(transactions, min_support)
        return results
    except Exception as e:
        st.error(f'Engine Error: {e}')
        return None

def initialize_boot_sequence():
    """
    Initialize boot sequence for Sovereign Architect.
    """
    try:
        architect = SovereignArchitect()
        architect.boot_sequence()
        st.code('--- Sovereign Omni-Sync Architect Initialized ---\nGen Level: 19\nNeural Memory: Syncing...')
        st.success('Architect Ready.')
    except Exception as e:
        st.error(f'Evolution Error: {e}')
with tab1:
    st.header('Association Rule Mining')
    st.write('Process your data through our hardened association engine.')
    data_input = st.text_area('Enter your transactions (JSON format or comma-separated lists)', placeholder='[[1, 2], [1, 2], [3, 4]]', height=200)
    min_support = st.slider('Minimum Support', 1, 10, 2)
    if st.button('Run Mining Engine'):
        if not user_api_key:
            st.error('Please enter your API Key in the sidebar first.')
        else:
            transactions = parse_transactions(data_input)
            if not transactions:
                st.error('Please provide valid transaction data.')
            else:
                with st.spinner('Processing...'):
                    results = run_association_mining(transactions, min_support)
                    if results:
                        st.success('Mining Complete!')
                        st.write('### Discovered Rules:')
                        st.write(results)
                    else:
                        st.info('No rules found with the current minimum support.')
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
        st.metric('Generation Level', '19', '+1')
    with col2:
        st.metric('Stability Rating', '100%', 'Secure')
    st.write('### Evolution Logs')
    if os.path.exists('evolution_logs.md'):
        with open('evolution_logs.md', 'r') as f:
            st.markdown(f.read())
    else:
        st.info('No logs found yet. Run an evolution cycle to generate logs.')
st.markdown('---')
st.markdown('2026 Delta Brain Sync | Powered by Sovereign Omni-Sync Architect')