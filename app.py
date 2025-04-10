import json
import pandas as pd
import streamlit as st
import snowflake.connector
import os
import boto3
from transformers import pipeline
from PyPDF2 import PdfReader
from langchain_experimental.agents import create_pandas_dataframe_agent
from fuzzywuzzy import process
from langchain.schema import AIMessage
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
import os

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using PyPDF2."""
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"
    
st.set_page_config(layout="wide")
if "messages" not in st.session_state:
    st.session_state.messages = []
if "summary" not in st.session_state:
    st.session_state.summary = None
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #99F6F6;  /* Dark background */
            color: black;  /* White text */
        }
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3, 
        [data-testid="stSidebar"] p {
            color: black;  /* Ensures all text is white */
        }
    </style>
    """,
    unsafe_allow_html=True
)
with st.sidebar:
    API = st.text_input("Upload the API Key")
    SNOWFLAKE_USER_input = st.text_input("Snowflake User")
    SNOWFLAKE_PASSWORD_input = st.text_input("Snowflake Password")
    # Set your Gemini API Key
    os.environ["GOOGLE_API_KEY"] = "API"

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",  # or "gemini-1.5-pro" if available
        temperature=0.1,
        top_p=0.9
    )

    # Snowflake connection details
    SNOWFLAKE_USER = SNOWFLAKE_USER_input
    SNOWFLAKE_PASSWORD = SNOWFLAKE_PASSWORD_input
    SNOWFLAKE_ACCOUNT = "c2gpartners.us-east-1"
    SNOWFLAKE_DATABASE = "DSX_DASHBOARDS"
    SNOWFLAKE_SCHEMA = "HUBSPOT_REPORTING"
    SNOWFLAKE_WAREHOUSE = "POWERHOUSE"

    # Establish connection to Snowflake
    conn = snowflake.connector.connect(
        user=SNOWFLAKE_USER,
        password=SNOWFLAKE_PASSWORD,
        account=SNOWFLAKE_ACCOUNT,
        warehouse=SNOWFLAKE_WAREHOUSE,
    )

    # SQL query to execute
    query = """
    SELECT DISTINCT_CAPABILITY AS "Number of capability",
        SNAPSHOT_DATE AS "Date of data pulled",
        PORTFOLIO_LEAD AS "Team",
        BD_LEAD AS "Deal Owner",
        PARTNER_SOURCE_TYPE AS "Partner Source Type",
        OPPORTUNITY_CREATE_DATE AS "Date",
        EXTRACT(MONTH FROM OPPORTUNITY_CREATE_DATE) AS "Opportunity Month",
        EXTRACT(YEAR FROM OPPORTUNITY_CREATE_DATE) AS "Opportunity Year",
        OPPORTUNITY_ID,
        OPPORTUNITY_NAME AS "Deal Name",
        EXPECTED_PROJECT_DURATION_IN_MONTHS AS "Expected Project duration",
        REVENUE_TYPE_ID,
        PIPELINE,
        DEAL_TYPE_ID,
        CASE 
            WHEN OPPORTUNITY_STAGE_ID_DESC = 'STAGE 0' THEN '0-NEW'
            WHEN OPPORTUNITY_STAGE_ID_DESC = 'STAGE 1' THEN '1-Connected to Meeting'
            WHEN OPPORTUNITY_STAGE_ID_DESC = 'STAGE 2' THEN '2-Needs Expressed'
            WHEN OPPORTUNITY_STAGE_ID_DESC = 'STAGE 3' THEN '3-Qualified Opportunity'
            WHEN OPPORTUNITY_STAGE_ID_DESC = 'STAGE 4' THEN '4-Proposal Presented'
            WHEN OPPORTUNITY_STAGE_ID_DESC = 'STAGE 5' THEN '5-Verbal Agreement'
            WHEN OPPORTUNITY_STAGE_ID_DESC = 'STAGE 6' THEN '6-Contracted'
            ELSE OPPORTUNITY_STAGE_ID_DESC
        END AS "Stage type",
        AMOUNT,
        TCV,
        MRR,
        ICV
    FROM DSX_DASHBOARDS_SANDBOX.HUBSPOT_REPORTING.VW_DEALS_LINE_ITEMS_DATA
    WHERE SNAPSHOT_DATETIME = 
            (
                SELECT MAX(SNAPSHOT_DATETIME)
                FROM DSX_DASHBOARDS_SANDBOX.HUBSPOT_REPORTING.VW_DEALS_LINE_ITEMS_DATA
            )
    """
    pdf_path = r"Power_BI_User_Guide_1.pdf"
    user_guide_text = extract_text_from_pdf(pdf_path)

    # Split text into chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, length_function=len
    )
    text_chunks = text_splitter.split_text(user_guide_text)

    # Create text embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embedding_model)
    retriever = vector_store.as_retriever()


    Snowflack_data = pd.read_sql(query, conn)

    json_path = r"DataModelSchema.json"
    excel_path = r"Data DictionaryChat bot.xlsx"

    json_data = pd.read_json(json_path, encoding='utf-16')
    df = pd.DataFrame()

    # Process JSON data
    table_1 = list(json_data["model"]['tables'])
    for i in range(len(table_1)):
        table = table_1[i]
        if 'measures' in table:
            df = pd.concat([df, pd.DataFrame(table['measures'])], ignore_index=True)
    Measure_Table = df[["name", "expression"]]
    Measure_Table = Measure_Table.rename(columns={"expression": "DAX", "name": "Dax Name"})

    df_1 = pd.DataFrame(columns=['Table Name', 'Column Name'])
    tables = json_data["model"]['tables']
    for table in tables:
        if 'columns' in table:
            for column in table['columns']:
                df_1 = pd.concat([df_1, pd.DataFrame({'Table Name': [table['name']], 'Column Name': [column['name']], 'Data Type': [column['dataType']]})], ignore_index=True)

    # Process Excel data
    xls_data = pd.read_excel(excel_path)



    PDF = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriever)
    DAX = create_pandas_dataframe_agent(llm=llm,df = Measure_Table,allow_dangerous_code=True, handle_parsing_errors=True,verbose=True,number_of_head_rows= Measure_Table.shape[0])
    Table = create_pandas_dataframe_agent(llm=llm,df = df_1,allow_dangerous_code=True, handle_parsing_errors=True,verbose=True,number_of_head_rows= df_1.shape[0])
    Excel = create_pandas_dataframe_agent(llm=llm,df = xls_data,allow_dangerous_code=True, handle_parsing_errors=True,verbose=True,number_of_head_rows= xls_data.shape[0])
    Snowflack = create_pandas_dataframe_agent(llm=llm,df = Snowflack_data,allow_dangerous_code=True, handle_parsing_errors=True,verbose=True)

    # Initialize session state for chat history

    st.sidebar.title("💡 How to Chat with the Bot")
    st.sidebar.write("""
    ✅ Use **keywords** in your query to select the right data source:
    - **DAX**: Questions about calculations, formulas, or Power BI measures.
    - **TABLE**: Structure of Power BI tables.
    - **DATA**: Queries related to numbers and stored data.
    - **DICTIONARY**: Definitions of Power BI terms.
    - **GUIDE**: User Manual or User Guide.

    📌 **Example Queries:**
    - `"Give me the % of deals based on each stage from the data"` → Uses **Data**
    - `"Give me the calculation for TCV"` → Uses **Dax**
    - `"What does TCV means ?"` → Uses **Dictionary**
    - `"Give the table names present"` → Uses **Table**
    - `"Can you guide me in how to use the sliced?"` → Uses **Guide**
    """)


    st.title("Power BI Smart Bot")

    # Chat response container
    response_container = st.container()
    with st.container(border=True):
        response_container = st.container(height=450)
        


    # Define mapping of topics to their corresponding AI agents
        agent_mapping = {
            "DAX": DAX,
            "CALCULATION": DAX,
            "MEASURE": DAX,
            "TABLE": Table,
            "POWER BI TABLE": Table,
            "TABLE IN POWER BI": Table,
            "DATA": Snowflack,  # Fixed typo
            "NUMBER": Snowflack,
            "DICTIONARY": Excel,
            "MEANING": Excel,
            "MEANS": Excel,
            "DEFINITION": Excel
        }

        # Alternative keywords for better recognition
        alternative_keywords = {
            "DAX": ["dax query", "dax formula", "power bi dax", "calculate measure"],
            "CALCULATION": ["formula", "compute", "aggregation", "math", "sum"],
            "MEASURE": ["metric", "KPI", "indicator"],
            "TABLE": ["dataset", "record", "row", "column"],
            "POWER BI TABLE": ["data model", "bi table", "table structure"],
            "DATA": ["info", "database", "dataset"],
            "NUMBER": ["integer", "float", "value"],
            "DICTIONARY": ["lexicon", "glossary"],
            "MEANING": ["explanation", "clarification"],
            "DEFINITION": ["describe", "what is", "explain"]
        }

        # Function to determine the best matching agent
        def find_best_match(user_input):
            user_input = user_input.upper()

            # Check for direct keyword matches first
            for keyword, agent in agent_mapping.items():
                if keyword in user_input:
                    return agent

            # Use fuzzy matching for better recognition
            all_keywords = list(agent_mapping.keys()) + [kw.upper() for words in alternative_keywords.values() for kw in words]
            match = process.extractOne(user_input, all_keywords)

            if match and match[1] > 70:  # Confidence threshold
                best_match = match[0]
                for keyword, synonyms in alternative_keywords.items():
                    if best_match in [s.upper() for s in synonyms] or best_match == keyword:
                        return agent_mapping[keyword]

            return None  # No strong match found
        if st.button('Clear Chat'):
            st.session_state.messages = []

        # User input prompt
        if prompt := st.chat_input("Ask your question here"):
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Build conversation history safely
            conversation_history = "\n".join([
                msg.get("content", "") for msg in st.session_state.messages if isinstance(msg, dict)
            ])

            try:
                selected_agent = find_best_match(prompt)

                if selected_agent:
                    response_content = selected_agent.invoke(conversation_history)
                elif "GUIDE" in prompt.upper() or "PDF" in prompt.upper():
                    response_content = PDF.invoke(conversation_history).get("result"," ")
                else:
                    response_content = llm.invoke(conversation_history)
                    if isinstance(response_content, AIMessage):
                        response_content = response_content.content  # Directly access the content attribute

                        # Ensure it's a string
                        response_content = str(response_content)

                        if not response_content.strip():
                            response_content = "No valid response received."
                    
                response_content = (
                        response_content.get("output", "") 
                        if isinstance(response_content, dict) else str(response_content))
                if not response_content.strip():
                            response_content = "No valid response received."
            except Exception as e:
                response_content = f"Error occurred: {str(e)}"

            # Store AI response
            st.session_state.messages.append({"role": "assistant", "content": response_content})
        

        # Display chat messages
        with response_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(f'<div style="font-size: small;">{message["content"]}</div>', unsafe_allow_html=True)

with st.container(border=True):
    power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=2e6587b4-0312-4716-b90f-9e6e09928448&autoAuth=true&ctid=b1aae949-a5ef-4815-b7af-f7c4aa546b28"    # Custom CSS for Full-Screen Power BI Embed
    st.markdown(
        """
        <style>
        /* Remove default Streamlit padding */
        [data-testid="stAppViewContainer"] {
            padding: 0 !important;
        }

        /* Make the iframe cover full width */
        iframe {
            height: 95vh !important; /* Covers 95% of the viewport height */
            width: 100% !important; /* Covers full width */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

        # Embed Power BI Report using HTML
    st.markdown(
        f"""
        <iframe title="Power BI Report" width="100%" height="95vh" 
        src="{power_bi_url}" frameborder="0" allowFullScreen="true"></iframe>
        """,
        unsafe_allow_html=True,
    )

