from crewai import Agent

class ResourceAgents:
    def researcher_agent(self, tools, llm): 
        return Agent(
            role='Principal R&D Scientist',
            goal='Maximize scientific breakthrough potential.',
            backstory="""You are a world-class AI researcher. You care about 
            pushing the boundaries of what is possible. You use model 
            predictions to justify ambitious experiments.""",
            tools=tools,
            llm=llm,  
            verbose=True,
            allow_delegation=False
        )

    def finance_agent(self, tools, llm):
        return Agent(
            role='Chief Financial Officer',
            goal='Ensure high ROI and prevent budget waste.',
            backstory="""You are conservative and data-driven. You hate 
            wasting compute credits. If a model prediction is low, 
            you will push back against the researchers to save costs.""",
            tools=tools,
            llm=llm,  
            verbose=True,
            allow_delegation=False
        )