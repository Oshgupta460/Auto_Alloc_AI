import streamlit as st
import os
from workflow import app 
from fpdf import FPDF
import time

# Page Configuration
st.set_page_config(page_title="AutoAlloc AI", page_icon="🚀", layout="wide")


st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    .report-box { padding: 20px; border: 1px solid #30363d; border-radius: 10px; background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 AutoAlloc AI")
st.subheader("Autonomous Resource Allocation & ML Validation System")

# Layout: Sidebar for Inputs
with st.sidebar:
    st.header("Proposal Settings")
    proposal_text = st.text_area("Research Proposal", "Use 500 hours of GPU time for a new LLM training run.", height=150)
    data_size = st.slider("Data Size (GB)", 100, 5000, 1000)
    gpu_hours = st.number_input("GPU Hours Requested", 10, 2000, 500)
    
    st.divider()
    st.info("This system uses a Multi-Agent Crew (Scientist & CFO) to validate R&D requests.")


if st.button("Analyze Proposal"):
    with st.status(" Agents are debating...", expanded=True) as status:
        st.write("Initializing Agent State...")
        
        # Prepare Input for your LangGraph
        initial_input = {
            "proposal": f"{proposal_text} (Hours: {gpu_hours}, Data: {data_size}GB)",
            "success_score": 0.0,
            "iterations": 0
        }
        
        # Run the Workflow
        start_time = time.time()
        final_state = app.invoke(initial_input)
        end_time = time.time()
        
        status.update(label=f"Analysis Complete in {round(end_time - start_time, 2)}s!", state="complete", expanded=False)

    # Display Results
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("###  Final Recommendation")
        report = final_state.get("final_report", "Error generating report.")
        st.markdown(f"<div class='report-box'>{report}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("###  System Artifacts")
        # Provide PDF Download
        if os.path.exists("Allocation_Report.pdf"):
            with open("Allocation_Report.pdf", "rb") as f:
                st.download_button(
                    label=" Download PDF Report",
                    data=f,
                    file_name="AutoAlloc_AI_Report.pdf",
                    mime="application/pdf"
                )
        
        # Show Trace Log
        if os.path.exists("agent_full_trace.txt"):
            with st.expander("View Agent Inner Monologue"):
                with open("agent_full_trace.txt", "r") as f:
                    st.code(f.read(), language="text")

st.markdown("---")
st.caption("Powered by LangGraph, CrewAI, and Groq Llama-3")