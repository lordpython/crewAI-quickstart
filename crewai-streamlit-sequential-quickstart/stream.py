import streamlit as st
import sys
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOpenAI
from textwrap import dedent
import re

# Define the StreamToStreamlit class
class StreamToStreamlit:
    def __init__(self, expander):
        self.expander = expander
        self.buffer = []
        self.colors = ['red', 'green', 'blue', 'orange']  # Define a list of colors
        self.color_index = 0  # Initialize color index

    def write(self, data):
        # Filter out ANSI escape codes using a regular expression
        cleaned_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', data)

        # Check if the data contains 'task' information
        task_match_object = re.search(r'\"task\"\s*:\s*\"(.*?)\"', cleaned_data, re.IGNORECASE)
        task_match_input = re.search(r'task\s*:\s*([^\n]*)', cleaned_data, re.IGNORECASE)
        task_value = None
        if task_match_object:
            task_value = task_match_object.group(1)
        elif task_match_input:
            task_value = task_match_input.group(1).strip()

        if task_value:
            st.info(":robot_face: " + task_value)

        # Check if the text contains the specified phrase and apply color
        if "Entering new CrewAgentExecutor chain" in cleaned_data:
            # Apply different color and switch color index
            self.color_index = (self.color_index + 1) % len(
                self.colors)  # Increment color index and wrap around if necessary

            cleaned_data = cleaned_data.replace("Entering new CrewAgentExecutor chain",
                                                f":{self.colors[self.color_index]}[Entering new CrewAgentExecutor chain]")

        if "Market Research Analyst" in cleaned_data:
            # Apply different color
            cleaned_data = cleaned_data.replace("Market Research Analyst",
                                                f":{self.colors[self.color_index]}[Market Research Analyst]")
        if "Business Development Consultant" in cleaned_data:
            cleaned_data = cleaned_data.replace("Business Development Consultant",
                                                f":{self.colors[self.color_index]}[Business Development Consultant]")
        if "Technology Expert" in cleaned_data:
            cleaned_data = cleaned_data.replace("Technology Expert",
                                                f":{self.colors[self.color_index]}[Technology Expert]")
        if "Finished chain." in cleaned_data:
            cleaned_data = cleaned_data.replace("Finished chain.", f":{self.colors[self.color_index]}[Finished chain.]")

        self.buffer.append(cleaned_data)
        if "\n" in data:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer = []

def main():
    st.sidebar.title('Customization')
    api = st.sidebar.selectbox(
        'Choose an API',
        ['Groq', 'OpenAI', 'Anthropic']
    )

    api_key = st.sidebar.text_input('Enter API Key', 'gsk-')

    temp = st.sidebar.slider("Model Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

    if api == 'Groq':
        model = st.sidebar.selectbox(
            'Choose a model',
            ['llama3-70b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
        )

        llm = ChatGroq(
            temperature=temp,
            model_name=model,
            groq_api_key=api_key
        )

    elif api == 'OpenAI':
        model = st.sidebar.selectbox(
            'Choose a model',
            ['gpt-4-turbo', 'gpt-4-1106-preview', 'gpt-3.5-turbo-0125']
        )

        llm = ChatOpenAI(
            temperature=temp,
            openai_api_key=api_key,
            model_name=model
        )

    elif api == 'Anthropic':
        model = st.sidebar.selectbox(
            'Choose a model',
            ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']
        )

        llm = ChatAnthropic(
            temperature=temp,
            anthropic_api_key=api_key,
            model_name=model
        )

    # Streamlit UI
    st.title('My New Crew')
    multiline_text = """
    This crew does something
    """
    st.markdown(multiline_text, unsafe_allow_html=True)

    # Display the CrewAI logo
    spacer, col = st.columns([5, 1])
    with col:
        st.image('crewai-logo.png')

    agent_template = dedent("""
        Defines the agent's function within the crew. It determines the kind of tasks the agent is best suited for.
    """)
    backstory_template = dedent("""
        Provides context to the agent's role and goal, enriching the interaction and collaboration dynamics.
    """)
    goal_template = dedent("""
        The individual objective that the agent aims to achieve. It guides the agent's decision-making process.
    """)

    agent_1 = Agent(
        role=agent_template,
        backstory=backstory_template,
        goal=goal_template,
        allow_delegation=False,
        verbose=True,
        max_iter=3,
        llm=llm,
    )

    agent_2 = Agent(
        role=agent_template,
        backstory=backstory_template,
        goal=goal_template,
        allow_delegation=False,
        verbose=True,
        max_iter=3,
        llm=llm,
    )

    agent_3 = Agent(
        role=agent_template,
        backstory=backstory_template,
        goal=goal_template,
        allow_delegation=False,
        verbose=True,
        max_iter=3,
        llm=llm,
    )

    var_1 = st.text_input("Variable 1:")
    var_2 = st.text_input("Variable 2:")
    var_3 = st.text_input("Variable 3:")

    if var_1 and var_2 and var_3 and api_key:
        if st.button("Start"):
            task_description_template = dedent(f"""
                A clear, concise statement of what the task entails.
                ---
                VARIABLE 1: "{var_1}"
                VARIABLE 2: "{var_2}"
                VARIABLE 3: "{var_3}"
                Add more variables if needed...
            """)
            expected_output_template = dedent("""
                A detailed description of what the task's completion looks like.
            """)

            task_1 = Task(
                description=task_description_template,
                expected_output=expected_output_template,
                agent=agent_1,
            )

            task_2 = Task(
                description=task_description_template,
                expected_output=expected_output_template,
                agent=agent_2,
                context=[task_1],
            )

            task_3 = Task(
                description=task_description_template,
                expected_output=expected_output_template,
                agent=agent_3,
                context=[task_2],
            )

            crew = Crew(
                agents=[agent_1, agent_2, agent_3],
                tasks=[task_1, task_2, task_3],
                verbose=2,
                process=Process.sequential
            )

            with st.spinner("Generating..."):
                expander = st.expander("Output", expanded=True)
                sys.stdout = StreamToStreamlit(expander)

                result = ""
                for delta in crew.kickoff():
                    result += delta  # Assuming delta is a string, if not, convert it appropriately
                    expander.markdown(result, unsafe_allow_html=True)

                # Reset stdout
                sys.stdout = sys.__stdout__

if __name__ == "__main__":
    main()
