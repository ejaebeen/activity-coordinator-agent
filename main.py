
def main():
    from activity_coordinator_agent.agents.context_agent import ContextAgent
    from activity_coordinator_agent.agents.idea_generation_agent import IdeaGenerationAgent
    from activity_coordinator_agent.models.context_model import Context

    agent = ContextAgent()
    context = agent.run("i have to create a fun 15 min activity for 5 elderly residents that has motor neurone disease. It is a craft activity")
    print(context)


    import asyncio
    idea_agent = IdeaGenerationAgent(n_ideas=3)
    ideas = asyncio.run(idea_agent.run_async(context=context, model_id="llama3.1"))
    print(ideas)

    from activity_coordinator_agent.agents.synthesis_agent import SynthesisAgent
    synthesis_agent = SynthesisAgent()
    final_answer = synthesis_agent.run(context=context, ideas=ideas)

    print(final_answer)


if __name__ == "__main__":
    main()
