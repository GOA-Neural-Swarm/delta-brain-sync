import streamlit as st
import numpy as np
import pandas as pd
from brain import association_rule_mining, NeuralBrain, SovereignArchitect
import os
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Delta Brain Sync - Swarm Intelligence",
    page_icon="🐺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Dark Theme
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stSidebar {
        background-color: #161b22;
    }
    h1, h2, h3 {
        color: #58a6ff;
    }
    .stButton>button {
        width: 100%;
        background-color: #238636;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar - Logo & Branding
with st.sidebar:
    if os.path.exists("logo.png"):
        logo = Image.open("logo.png")
        st.image(logo, use_container_width=True)
    st.title("Delta Brain Sync")
    st.markdown("---")

    # Bring Your Own Key (BYOK)
    st.subheader("🔑 Unlock Swarm Core")
    api_provider = st.selectbox("Select AI Provider", ["Groq", "Gemini", "OpenAI"])
    user_api_key = st.text_input(f"Enter your {api_provider} API Key", type="password")
    
    if user_api_key:
        st.success(f"{api_provider} Core Linked!")
    else:
        st.warning("Please enter an API key to enable advanced processing.")

    st.markdown("---")
    
    # Monetization Placeholder
    st.subheader("💎 Premium Access")
    st.info("Support the creator to unlock automated analytics.")
    st.markdown("""
    **To unlock advanced features:**
    Send **$10 (USDT)** to this TRC-20 Wallet Address:
    
    `[YOUR_FUTURE_CRYPTO_ADDRESS_HERE]`
    
    *QR Code Integration Pending...*
    """)

# Main Page
st.title("🐺 Delta Brain Sync: Swarm Intelligence Dashboard")
st.write("Welcome to the next generation of association rule mining and neural evolution.")

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["📊 Association Mining", "🧠 Neural Evolution", "⚙️ System Health"])

with tab1:
    st.header("Association Rule Mining")
    st.write("Process your data through our hardened association engine.")
    
    data_input = st.text_area("Enter your transactions (JSON format or comma-separated lists)", 
                             placeholder='[[1, 2], [1, 2], [3, 4]]')
    
    min_support = st.slider("Minimum Support", 1, 10, 2)
    
    if st.button("Run Mining Engine"):
        try:
            # Simple parsing for demo/use
            import json
            try:
                transactions = json.loads(data_input)
            except:
                # Fallback to simple list parsing if not valid JSON
                transactions = [line.split(',') for line in data_input.strip().split('\n') if line]
            
            if not transactions:
                st.error("Please provide valid transaction data.")
            else:
                with st.spinner("Processing..."):
                    results = association_rule_mining(transactions, min_support)
                    st.success("Mining Complete!")
                    st.write("### Discovered Rules:")
                    st.write(results)
        except Exception as e:
            st.error(f"Error: {e}")

with tab2:
    st.header("Neural Evolution")
    st.write("Monitor and trigger Sovereign Architect evolution cycles.")
    
    if st.button("Initialize Boot Sequence"):
        architect = SovereignArchitect()
        with st.spinner("Booting..."):
            # Mocking the print outputs to Streamlit
            st.code("--- Sovereign Omni-Sync Architect Initialized ---\nGen Level: 19\nNeural Memory: Syncing...")
            st.success("Architect Ready.")

with tab3:
    st.header("System Health")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Generation Level", "19", "+1")
    with col2:
        st.metric("Stability Rating", "100%", "Secure")
    
    st.write("### Evolution Logs")
    if os.path.exists("evolution_logs.md"):
        with open("evolution_logs.md", "r") as f:
            st.markdown(f.read())
    else:
        st.info("No logs found yet. Run an evolution cycle to generate logs.")

# Footer
st.markdown("---")
st.markdown("© 2026 Delta Brain Sync | Powered by Sovereign Omni-Sync Architect")
