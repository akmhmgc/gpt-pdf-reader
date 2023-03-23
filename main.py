import tempfile
import base64
from pathlib import Path
from llama_index import download_loader
import streamlit as st
from langchain.chat_models import ChatOpenAI
from llama_index import GPTSimpleVectorIndex, LLMPredictor, download_loader

def show_pdf(file_path:str):
    """Show the PDF in Streamlit
    That returns as html component

    Parameters
    ----------
    file_path : [str]
        Uploaded PDF file path
    """

    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="1000" type="application/pdf">'
    return pdf_display


st.title("PDF Indexer")

st.write("Enter your OpenAI API key")
openai_api_key = st.text_input("OpenAI API Key", value="", type="password")

if 'index' not in st.session_state:
    st.session_state['index'] = None

if openai_api_key:
    st.write("API keys have been set.")

    uploaded_file = st.file_uploader("Upload pdf", type="pdf")

    if uploaded_file and st.button("Make index data"):
        with st.spinner("Loading data..."):
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                fp = Path(tmp_file.name)
                fp.write_bytes(uploaded_file.getvalue())
                st.write(show_pdf(tmp_file.name))

                PDFReader = download_loader("PDFReader")
                loader = PDFReader()
                documents = loader.load_data(file=fp)
                st.success("Data loaded successfully!")
                print(documents)

                llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key))
                st.session_state.index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor)

    if st.session_state.index:
        user_question = st.text_input("Enter your question:", value="")
        if user_question:
            output = st.session_state.index.query(user_question)
            st.write("Response:")
            st.markdown(f"<h3 style='font-size: 18px;'>{output}</h3>", unsafe_allow_html=True)

            st.write("Source:")
            st.markdown(f"<h3 style='font-size: 18px;'>{output.source_nodes[0].source_text[:100]}...</h3>", unsafe_allow_html=True)
