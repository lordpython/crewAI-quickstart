import re
import streamlit as st

class StreamToStreamlit:
    def __init__(self, container):
        self.container = container
        self.buffer = []
        self.colors = {
            'Visionary Innovator': 'violet',
            'Knowledge Explorer': 'blue',
            'Creative Technologist': 'green',
            'System': 'gray'
        }

    def write(self, data):
        cleaned_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', data)

        # Check for task information
        task_match = re.search(r'Task\s*:\s*([^\n]*)', cleaned_data, re.IGNORECASE)
        if task_match:
            task_value = task_match.group(1).strip()
            self.container.info(f"🎯 New Task: {task_value}")

        # Identify the agent or system message
        agent_match = re.search(r'(Visionary Innovator|Knowledge Explorer|Creative Technologist|Human Input Required):', cleaned_data)
        if agent_match:
            agent = agent_match.group(1)
            color = self.colors.get(agent, 'gray')
            icon = '🧠' if agent != 'Human Input Required' else '👤'
            cleaned_data = re.sub(f'{agent}:', f":{color}[{icon} {agent}:]", cleaned_data)
        elif "Entering new CrewAgentExecutor chain" in cleaned_data:
            cleaned_data = cleaned_data.replace("Entering new CrewAgentExecutor chain", ":gray[🔄 Starting new process...]")
        elif "Finished chain." in cleaned_data:
            cleaned_data = cleaned_data.replace("Finished chain.", ":gray[✅ Process completed.]")

        # Handle human input requests
        if "Human Input Required:" in cleaned_data:
            question = cleaned_data.split("Human Input Required:")[-1].strip()
            user_input = self.container.text_input("👤 Human Input Required:", key=f"human_input_{len(self.buffer)}")
            if user_input:
                cleaned_data += f"\nHuman response: {user_input}"

        self.buffer.append(cleaned_data)
        if "\n" in data or len(self.buffer) > 5:  # Adjust buffer size as needed
            self.container.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer = []

    def flush(self):
        if self.buffer:
            self.container.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer = []
