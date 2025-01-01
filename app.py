import os
import re
import streamlit as st
from io import BytesIO
from docx import Document
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
import sys

class StreamToExpander:
    def __init__(self, expander, buffer_limit=10000):
        self.expander = expander
        self.buffer = []
        self.buffer_limit = buffer_limit

    def write(self, data):
        # Clean ANSI escape codes from output
        cleaned_data = re.sub(r'\x1B\[\d+;?\d*m', '', data)
        if len(self.buffer) >= self.buffer_limit:
            self.buffer.pop(0)
        self.buffer.append(cleaned_data)

        if "\n" in data:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer.clear()

    def flush(self):
        if self.buffer:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer.clear()      

class ProposalWriter:
    def __init__(self):
        
        self.search_tool = SerperDevTool()
         
    def setup_agents(self):
        # Define Agents
        researcher = Agent(
            role='Research Analyst',
            goal='Analyze input documents and gather relevant context.',
            backstory='Expert at analyzing documents and extracting key information.',
            tools=[self.search_tool],
            allow_delegation=True,
            verbose=True
        )

        writer = Agent(
            role='Proposal Writer',
            goal='Create compelling and professional proposals.',
            backstory='Experienced in creating persuasive content.',
            tools=[self.search_tool],
            allow_delegation=True,
            verbose=True
        )

        editor = Agent(
            role='Editor',
            goal='Polish and refine proposal content.',
            backstory='Senior editor ensuring professional content standards.',
            tools=[self.search_tool],
            allow_delegation=True,
            verbose=True
        )

        return researcher, writer, editor

    def create_tasks(self, researcher, writer, editor, content):
        # Define Tasks
        research_task = Task(
            description=f"Analyze the following content and identify key points for the proposal:\n{content}",
            agent=researcher,
            expected_output="A detailed analysis including key themes, requirements, opportunities, and relevant market context."
        )

        writing_task = Task(
            description="Using research insights, create a professional proposal.",
            agent=writer,
            expected_output="A complete, well-structured proposal document with all required sections."
        )

        editing_task = Task(
            description="Review and refine the proposal for clarity, coherence, and professional tone.",
            agent=editor,
            expected_output="A polished, error-free proposal document."
        )

        return [research_task, writing_task, editing_task]

    def extract_text_from_docx(self, file_bytes):
        # Extract text from uploaded DOCX file
        doc = Document(BytesIO(file_bytes))
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])

    # Save results to file and provide download link
      
def save_and_provide_download_link(results):
    try:
        filename = "Proposal.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(str(results))
        
        with open(filename, "rb") as f:
            st.download_button(
                label="Download Proposal",
                data=f,
                file_name=filename,
                mime="text/plain"
            )
        if os.path.exists(filename):
            os.remove(filename)
    except IOError as e:
        st.error(f"File I/O error: {e}")
    except Exception as e:
        st.error(f"Failed to save and provide download link: {e}")

def main():
    # Configure Streamlit
    st.set_page_config(page_title="AI Proposal Writer", layout="wide")
    st.title("🚀 AI-Powered Proposal Writer")
    st.markdown("---")

    # Initialize ProposalWriter
    proposal_writer = ProposalWriter()
    api_key = st.text_input("OpenAI API Key", type="password")

    # File Upload
    uploaded_file = st.file_uploader("Upload your document (.docx)", type=['docx'])

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        if uploaded_file and st.button("Generate Proposal"):
            process_output_expander = st.expander("Processing Output:")
            sys.stdout = StreamToExpander(process_output_expander)
            with st.spinner("Processing your document..."):
                try:
                    # Extract text from the uploaded document
                    file_bytes = uploaded_file.read()
                    content = proposal_writer.extract_text_from_docx(file_bytes)

                    # Set up agents and tasks
                    researcher, writer, editor = proposal_writer.setup_agents()
                    tasks = proposal_writer.create_tasks(researcher, writer, editor, content)

                    # Run Crew
                    crew = Crew(
                        agents=[researcher, writer, editor],
                        tasks=tasks,
                        process=Process.sequential,
                        verbose=True
                    )
                    result = crew.kickoff()

                    # Display generated proposal
                    st.markdown("### Generated Proposal")
                    st.text_area("Proposal Content", result, height=300)

                    save_and_provide_download_link(result)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.error("Please try again or contact support if the issue persists.")

    else:
        st.error("Please provide your OpenAI API key.")    

if __name__ == "__main__":
    main()