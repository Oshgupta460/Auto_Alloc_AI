import os
from dotenv import load_dotenv
from workflow import app
from fpdf import FPDF
import litellm

load_dotenv()

def save_as_pdf(full_interaction, summary_report):
    """
    Creates a professional PDF containing both the 
    Full Agent Conversation and the Final Summary.
    """
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # --- Section 1: Final Summary ---
        pdf.set_font("Helvetica", 'B', 16)
        pdf.cell(0, 10, "Final Allocation Report", ln=True, align='C')
        pdf.ln(5)
        
        pdf.set_font("Helvetica", size=12)
        # Clean text for FPDF compatibility
        clean_summary = str(summary_report).encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 10, txt=clean_summary)
        
        pdf.ln(10)
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(10)

        # --- Section 2: Full Conversation Logs ---
        pdf.set_font("Helvetica", 'B', 14)
        pdf.cell(0, 10, "Full Agent Trace (Inner Monologue)", ln=True)
        pdf.ln(5)
        
        pdf.set_font("Courier", size=10) 
        clean_trace = str(full_interaction).encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 8, txt=clean_trace)

        pdf.output("Allocation_Report.pdf")
        print("\n PDF Generated: Allocation_Report.pdf")
    except Exception as e:
        print(f" PDF Error: {e}")

def run_system():
    print(" Starting Autonomous Allocation System...")
    
    # 2. Define the initial state
    initial_input = {
        "proposal": "Use 500 hours of GPU time for LLM training run.",
        "success_score": 0.0,
        "iterations": 0
    }

    # 3. Execute the Graph
    print(" AI Agents are thinking... (Check terminal for live logs)")
    final_state = app.invoke(initial_input)

    # 4. Gather Data for the Report
    # Pull the final answer
    summary = final_state.get("final_report", "No summary generated.")
    
    # Pull the full trace 
    full_trace = "No trace file found."
    if os.path.exists("agent_full_trace.txt"):
        with open("agent_full_trace.txt", "r", encoding="utf-8") as f:
            full_trace = f.read()

    # 5. Final Terminal Printout
    print("\n" + "="*50)
    print(" FINAL SYSTEM OUTPUT")
    print("="*50)
    print(summary)
    
    # 6. Save to Files
    with open("final_report.txt", "w", encoding="utf-8") as f:
        f.write(f"--- FINAL SUMMARY ---\n{summary}\n\n--- FULL TRACE ---\n{full_trace}")

    save_as_pdf(full_trace, summary)

if __name__ == "__main__":
    run_system()

    litellm.set_verbose = False
litellm.suppress_debug_info = True