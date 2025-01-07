import os
import re
import streamlit as st
from io import BytesIO
from docx import Document
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
# from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
import tempfile
import sys

class StreamToExpander:
    def __init__(self, expander, buffer_limit=10000):
        self.expander = expander
        self.buffer = []
        self.buffer_limit = buffer_limit

    def write(self, data):
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
        researcher = Agent(
            role='Research Analyst',
            goal='Analyze input documents and past project accomplishments to identify relevant experience.',
            backstory='Expert at analyzing documents and matching past performance to new requirements.',
            tools=[self.search_tool],
            allow_delegation=True,
            verbose=True
        )

        writer = Agent(
            role='Proposal Writer',
            goal='Create compelling proposals that highlight relevant past performance.',
            backstory='Experienced in creating persuasive content that demonstrates capability through past accomplishments.',
            tools=[self.search_tool],
            allow_delegation=True,
            verbose=True
        )

        editor = Agent(
            role='Editor',
            goal='Polish proposal content while ensuring past performance claims are well-supported.',
            backstory='Senior editor ensuring professional standards and effective past performance narratives.',
            tools=[self.search_tool],
            allow_delegation=True,
            verbose=True
        )

        return researcher, writer, editor

    def create_tasks(self, researcher, writer, editor, content, past_performance_summary):
        research_task = Task(
            description=f"""Analyze the following content and past performance documents to identify key points and relevant experience:
            
            New Proposal Requirements:
            {content}
            
            Past Performance Summary:
            {past_performance_summary}
            
            Identify specific past projects and accomplishments that align with each requirement.""",
            agent=researcher,
            expected_output="A detailed analysis mapping past performance to new requirements, including specific examples and outcomes."
        )

        writing_task = Task(
            description="""Using research insights, create a professional proposal that includes:
            1. Main proposal sections
            2. Specific paragraphs for each scope item that reference relevant past performance
            3. Clear connections between past accomplishments and current requirements""",
            agent=writer,
            expected_output="A complete proposal with targeted past performance references for each requirement."
        )

        editing_task = Task(
            description="Review and refine the proposal, ensuring past performance references are compelling and well-integrated.",
            agent=editor,
            expected_output="A polished proposal that effectively demonstrates capability through past performance."
        )

        return [research_task, writing_task, editing_task]

    def process_documents(self, files):
        """Process multiple documents and create summaries using LangChain."""
        summaries = []
        
        for file in files:
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, file.name)
            
            try:
                with open(temp_path, 'wb') as tmp_file:
                    tmp_file.write(file.getvalue())
                
                loader = Docx2txtLoader(temp_path)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=200
                )
                split_docs = text_splitter.split_documents(documents)
                
                llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
                chain = load_summarize_chain(llm, chain_type="map_reduce")
                summary = chain.run(split_docs)
                summaries.append(f"Summary of {file.name}:\n{summary}\n")
                
            finally:
                # Clean up temporary files
                try:
                    os.remove(temp_path)
                    os.rmdir(temp_dir)
                except:
                    pass
        
        return "\n".join(summaries)

def save_and_provide_download_link(results):
    try:
        filename = "Proposal.docx"
        doc = Document()
        
        # Add title
        doc.add_heading('Project Proposal', 0)
        
        # Split the results into sections and add to document
        sections = str(results).split('\n\n')
        for section in sections:
            if section.strip():
                # Check if it's a heading (you can modify this logic based on your needs)
                if len(section.strip()) < 100 and section.isupper():
                    doc.add_heading(section, level=1)
                else:
                    doc.add_paragraph(section)
        
        # Save to BytesIO object
        doc_io = BytesIO()
        doc.save(doc_io)
        doc_io.seek(0)
        
        # Create download button
        st.download_button(
            label="Download Proposal",
            data=doc_io,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        
    except Exception as e:
        st.error(f"Failed to save and provide download link: {e}")

def main():
    st.set_page_config(page_title="AI Proposal Writer", layout="wide")
    st.title("🚀 AI-Powered Proposal Writer")
    st.markdown("---")

    proposal_writer = ProposalWriter()
    api_key = st.text_input("OpenAI API Key", type="password")

    # Main RFP document upload
    st.subheader("Upload Main RFP Document")
    main_file = st.file_uploader("Upload your RFP document (.docx)", type=['docx'])

    # Past performance documents upload
    st.subheader("Upload Past Performance Documents")
    past_performance_files = st.file_uploader(
        "Upload past performance documents (.docx)",
        type=['docx'],
        accept_multiple_files=True
    )

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        
        if main_file and past_performance_files and st.button("Generate Proposal"):
            process_output_expander = st.expander("Processing Output:")
            sys.stdout = StreamToExpander(process_output_expander)
            
            with st.spinner("Processing documents..."):
                try:
                    # Process main RFP document
                    main_content = Document(BytesIO(main_file.read()))
                    main_text = '\n'.join([paragraph.text for paragraph in main_content.paragraphs])

                    # Process past performance documents
                    st.info("Summarizing past performance documents...")
                    past_performance_summary = proposal_writer.process_documents(past_performance_files)

                    # Set up and run the crew
                    researcher, writer, editor = proposal_writer.setup_agents()
                    tasks = proposal_writer.create_tasks(
                        researcher,
                        writer,
                        editor,
                        main_text,
                        past_performance_summary
                    )
                    st.info("Generating proposal...")
                    crew = Crew(
                        agents=[researcher, writer, editor],
                        tasks=tasks,
                        process=Process.sequential,
                        verbose=True
                    )
                    result = crew.kickoff()

                    # Display results
                    st.markdown("### Generated Proposal")
                    st.text_area("Proposal Content", result, height=300)
                    
                    # Provide download option
                    save_and_provide_download_link(result)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.error("Please try again or contact support if the issue persists.")

    else:
        st.error("Please provide your OpenAI API key.")

if __name__ == "__main__":
    main()