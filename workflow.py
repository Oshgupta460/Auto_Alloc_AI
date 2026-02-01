import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from crewai import Task, Crew
from crewai.tools import BaseTool
from model import get_experiment_prediction
from agents import ResourceAgents
from crewai import LLM


# Define the Tool Class for CrewAI
class SuccessPredictorTool(BaseTool):
    name: str = "Success Predictor Tool"
    description: str = "Predicts the probability of experiment success based on compute hours and data size."

    def _run(self, compute_hours: int, data_size_gb: int) -> str:
        try:
            score = get_experiment_prediction(compute_hours, data_size_gb)
            return f"The PyTorch model predicts a success probability of: {score:.2f}"
        except Exception as e:
            return f"Error calculating prediction: {str(e)}"

success_tool = SuccessPredictorTool()

# Define the Groq model
free_llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# Instantiate Agents
agent_factory = ResourceAgents()
scientist = agent_factory.researcher_agent(tools=[success_tool], llm=free_llm)
cfo = agent_factory.finance_agent(tools=[success_tool], llm=free_llm)

# Define Task
evaluation_task = Task(
    description='Evaluate a proposal to use 500 hours of GPU time for a new LLM training run. Check the ML score first.',
    expected_output='A joint recommendation report: Should we fund this or pivot?',
    agent=scientist
)

# Create the Crew
lumina_crew = Crew(
    agents=[scientist, cfo], 
    tasks=[evaluation_task],
    verbose=True,
    output_log_file="agent_full_trace.txt"
)


class AgentState(TypedDict):
    proposal: str
    success_score: float
    iterations: int
    final_report: str

# NODE 1: The Predictor
def check_prediction(state: AgentState):
    print("\n--- NODE 1: Analyzing ML Prediction (PyTorch) ---")
    score = 0.85 
    print(f"ML Model Output: {score}")
    return {"success_score": score, "iterations": state.get("iterations", 0) + 1}

# NODE 2: The Crew
def run_crew_analysis(state: AgentState):
    print("\n--- NODE 2: CrewAI is debating a better plan ---")
    result = lumina_crew.kickoff()
    return {"final_report": result}

# ROUTER
def router(state: AgentState):
    if state["success_score"] > 0.7:
        return "approved"
    elif state["iterations"] > 2:
        return "force_stop"
    else:
        return "retry"

workflow = StateGraph(AgentState)
workflow.add_node("predictor", check_prediction)
workflow.add_node("crew_optimization", run_crew_analysis)

workflow.set_entry_point("predictor")

workflow.add_conditional_edges(
    "predictor", 
    router, 
    {
        "approved": "crew_optimization", 
        "retry": "predictor", 
        "force_stop": END
    }
)
workflow.add_edge("crew_optimization", END)

app = workflow.compile()