# 🚀 AutoAlloc AI 

**Multi-Agent Resource Orchestration & ML Validation Framework**

AutoAlloc AI is an agentic workflow designed to automate the approval process for expensive R&D resources (like GPU compute). It combines deep learning predictions with multi-agent debate to provide data-driven, audited allocation decisions.

##  How it Works
1. **ML Validation:** A PyTorch model predicts the success probability of a proposal based on historical data.
2. **Agentic Debate:** A **Principal Scientist** agent and a **CFO** agent (powered by CrewAI & Groq) debate the merits of the proposal.
3. **Structured Output:** The system generates a formal PDF recommendation report via a professional Streamlit dashboard.

##  Tech Stack
- **Frameworks:** LangGraph, CrewAI
- **Brain:** Groq (Llama-3-70b)
- **ML Logic:** PyTorch
- **UI:** Streamlit
- **Reporting:** FPDF2

##  Quick Start
1. Clone the repo: `git clone https://github.com/Oshgupta460/AutoAlloc-AI.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Add your `GROQ_API_KEY` to a `.env` file.
4. Run the app: `streamlit run app.py`

---
*Developed as a showcase of Agentic Workflows and LLM Orchestration.*