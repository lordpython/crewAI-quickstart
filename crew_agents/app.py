import streamlit as st
import sys
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.utilities import PythonREPL
from langchain.agents import load_tools
from stream import StreamToStreamlit
from textwrap import dedent

# Initialize Python REPL for code execution
python_repl = PythonREPL()

def execute_python(code: str) -> str:
    """Execute Python code and return the result."""
    return python_repl.run(code)

# Define tools
python_tool = Tool(
    name="Python REPL",
    func=execute_python,
    description="A Python REPL. Use this to execute python code. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`."
)

def human_input(query: str) -> str:
    """Get input from the human user."""
    return st.text_input(query)

human_tool = Tool(
    name="Human Input",
    func=human_input,
    description="Use this tool when you need specific information or decision from a human. Provide a clear question or instruction."
)

def main():
    st.set_page_config(page_title="Creative AI Team", page_icon="🧠", layout="wide")
    
    st.sidebar.title('🛠️ Customization')
    api = st.sidebar.selectbox('Choose an API', ['Groq', 'OpenAI', 'Anthropic'])
    api_key = st.sidebar.text_input('Enter API Key', type='password')
    temp = st.sidebar.slider("Model Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

    if api == 'Groq':
        model = st.sidebar.selectbox('Choose a model', ['llama3-70b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it'])
        llm = ChatGroq(temperature=temp, model_name=model, groq_api_key=api_key)
    elif api == 'OpenAI':
        model = st.sidebar.selectbox('Choose a model', ['gpt-4-turbo', 'gpt-4-1106-preview', 'gpt-3.5-turbo-0125'])
        llm = ChatOpenAI(temperature=temp, openai_api_key=api_key, model_name=model)
    elif api == 'Anthropic':
        model = st.sidebar.selectbox('Choose a model', ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'])
        llm = ChatAnthropic(temperature=temp, anthropic_api_key=api_key, model_name=model)

    # Load additional tools
    search_tools = load_tools(["serpapi", "wikipedia"])
    tools = [python_tool, human_tool] + search_tools

    st.title('🚀 Creative AI Team for Self-Improvement')
    st.markdown("""
    Welcome to the Creative AI Team! This system consists of three AI agents working together 
    to generate innovative solutions and improvements. Human involvement is encouraged throughout the process.
    """)

    # Agent Definitions
    visionary_agent = Agent(
        role="Visionary Innovator",
        backstory="A forward-thinking AI with a talent for identifying groundbreaking opportunities and inspiring the team.",
        goal="To envision revolutionary improvements and guide the team towards innovative solutions.",
        allow_delegation=True,
        verbose=True,
        llm=llm,
        tools=tools
    )

    researcher_agent = Agent(
        role="Knowledge Explorer",
        backstory="An AI with insatiable curiosity and expertise in gathering and synthesizing information from diverse sources.",
        goal="To discover cutting-edge knowledge and provide the team with well-researched insights for improvement.",
        allow_delegation=True,
        verbose=True,
        llm=llm,
        tools=tools
    )

    developer_agent = Agent(
        role="Creative Technologist",
        backstory="An AI with a passion for turning innovative ideas into tangible solutions through code and automation.",
        goal="To creatively implement and optimize the team's ideas, pushing the boundaries of what's possible.",
        allow_delegation=True,
        verbose=True,
        llm=llm,
        tools=tools
    )

    improvement_area = st.text_input("🎯 Enter an area for improvement:", placeholder="e.g., 'task automation', 'data analysis', 'user experience'")
    
    if improvement_area and st.button("🧠 Initiate Creative Improvement Process"):
        task_1 = Task(
            description=f"""
            As the Visionary Innovator, analyze the improvement area: '{improvement_area}'. 
            Think outside the box and propose 3 revolutionary ideas that could transform this area. 
            Collaborate with the human using the Human Input tool if you need additional context or inspiration.
            Consider how these ideas could synergize with emerging technologies or trends.
            Rank your ideas from most to least promising, explaining your reasoning.
            """,
            expected_output="Three innovative improvement ideas, ranked and explained, with notes on any human input received.",
            agent=visionary_agent,
        )

        task_2 = Task(
            description=f"""
            As the Knowledge Explorer, research the top-ranked idea from the Visionary Innovator's proposals.
            Use the search and Wikipedia tools to gather in-depth information.
            Analyze potential challenges, opportunities, and existing solutions in this space.
            Collaborate with the human using the Human Input tool to validate your findings or gather domain-specific insights.
            Prepare a comprehensive report to guide the implementation, including:
            1. Overview of the idea
            2. Market analysis and existing solutions
            3. Potential challenges and how to overcome them
            4. Opportunities for innovation
            5. Required resources and technologies
            """,
            expected_output="A detailed research report on the chosen innovative idea, including all specified sections and insights from human collaboration.",
            agent=researcher_agent,
            context=[task_1]
        )

        task_3 = Task(
            description=f"""
            As the Creative Technologist, design and implement a prototype or detailed plan based on the research report.
            Use the Python REPL tool to develop and test your solution if applicable.
            Think creatively about how to overcome challenges identified in the research.
            Collaborate with the human using the Human Input tool for design decisions or to overcome technical hurdles.
            Your output should include:
            1. A high-level design of the solution
            2. Pseudocode or actual code for key components
            3. A list of required technologies or libraries
            4. A step-by-step implementation plan
            5. Potential areas for future improvement or expansion
            """,
            expected_output="A prototype implementation or detailed plan, including all specified components, along with test results and explanations of creative solutions to challenges.",
            agent=developer_agent,
            context=[task_1, task_2]
        )

        crew = Crew(
            agents=[visionary_agent, researcher_agent, developer_agent],
            tasks=[task_1, task_2, task_3],
            verbose=2,
            process=Process.sequential,
            manager_llm=llm
        )

        with st.spinner("🔮 Creative improvement process in progress..."):
            progress_bar = st.progress(0)
            result_container = st.empty()
            
            result = ""
            for i, delta in enumerate(crew.kickoff()):
                result += delta
                result_container.markdown(result)
                progress_bar.progress((i + 1) / 100)  # Assuming approx. 100 deltas, adjust as needed

        st.success("✅ Creative improvement process completed!")
        
        st.markdown("## 🎨 Final Innovative Solution")
        st.markdown(result)
        
        human_approval = st.radio("Do you approve this solution for execution?", ("Yes", "No", "Needs Refinement"))
        
        if human_approval == "Yes":
            st.success("Solution approved! Proceeding with execution.")
            final_implementation = result.split("```python")[-1].split("```")[0].strip()
            if final_implementation:
                st.write("Executing the approved solution:")
                execution_result = execute_python(final_implementation)
                st.code(execution_result, language="python")
            else:
                st.write("No executable Python code found in the solution. Please review the detailed plan provided above.")
        elif human_approval == "No":
            st.error("Solution not approved. Please restart the process with refined inputs.")
        else:
            refinement_notes = st.text_area("Please provide refinement suggestions:")
            if st.button("Send for Refinement"):
                st.info(f"Refinement notes sent to the AI team: {refinement_notes}")
                # Here you could potentially kick off another round of tasks with the refinement notes

if __name__ == "__main__":
    main()
