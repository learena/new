import streamlit as st
from langchain_community.document_loaders import TextLoader
from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel
from dotenv import load_dotenv
import os

load_dotenv()

# Function to read PDF files
def read_pdf(file):
    document = ""
    reader = PdfReader(file)
    for page in reader.pages:
        document += page.extract_text()
    return document

# Function to read TXT files
def read_txt(file):
    document = str(file.getvalue())
    document = document.replace("\\n", " \\n ").replace("\\r", " \\r ")
    return document

# Function to split documents into chunks
def split_doc(document, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split = splitter.split_text(document)
    split = splitter.create_documents(split)
    return split

# Function to create or update embeddings and store in FAISS
def embedding_storing(split, create_new_vs, existing_vector_store, new_vs_name):
    if create_new_vs is not None:
        instructor_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/distiluse-base-multilingual-cased-v1",
            model_kwargs={'device': 'cpu'}
        )

        # Create or update embeddings
        db = FAISS.from_documents(split, instructor_embeddings)

        if create_new_vs:
            # Save new vector store
            db.save_local("vector store/" + new_vs_name)
        else:
            # Load and merge with existing vector store
            load_db = FAISS.load_local(
                "vector store/" + existing_vector_store,
                instructor_embeddings,
                allow_dangerous_deserialization=True
            )
            load_db.merge_from(db)
            load_db.save_local("vector store/" + new_vs_name)

        st.success("The document has been saved.")

# Function to prepare Fauno-Italian-LLM-7B for conversational retrieval
def prepare_rag_llm(token, vector_store_list, temperature, max_length):
    # Load tokenizer and Fauno model
    tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
    base_model = LlamaForCausalLM.from_pretrained(
        "decapoda-research/llama-7b-hf",
        load_in_8bit=True,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, "andreabac3/Fauno-Italian-LLM-7B")
    model.eval()

    # Load vector store
    instructor_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/distiluse-base-multilingual-cased-v1",
        model_kwargs={'device': 'cpu'}
    )
    loaded_db = FAISS.load_local(
        f"vector store/{vector_store_list}",
        instructor_embeddings,
        allow_dangerous_deserialization=True
    )

    memory = ConversationBufferWindowMemory(
        k=2,
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )

    # Define response generation function
    def generate_response(question: str) -> dict:
        prompt = f"The conversation between human and AI assistant.\n[|Human|] {question}.\n[|AI|] "
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        generation_output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_length,
            temperature=temperature
        )
        output = tokenizer.decode(generation_output[0]).split("[|AI|]")[1]
        return {"answer": output}

    # Create a conversational chain
    qa_conversation = ConversationalRetrievalChain.from_llm(
        llm=generate_response,
        chain_type="stuff",
        retriever=loaded_db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        memory=memory,
    )

    return qa_conversation

# Function to generate answers
def generate_answer(question, token):
    if token == "":
        return "Insert the Hugging Face token", ["no source"]
    else:
        # Generate response using the conversation chain
        response = st.session_state.conversation({"question": question})
        answer = response.get("answer").strip()
        explanation = response.get("source_documents", [])
        doc_source = [d.page_content for d in explanation]
        return answer, doc_source

# Streamlit app main logic
st.title("Conversational AI with Fauno-Italian-LLM-7B")

# Upload documents
uploaded_file = st.file_uploader("Upload a document (PDF or TXT)", type=["pdf", "txt"])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        document = read_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        document = read_txt(uploaded_file)
    else:
        st.error("Unsupported file type.")

    # Split and store embeddings
    split_docs = split_doc(document, chunk_size=1000, chunk_overlap=200)
    create_new = st.checkbox("Create a new vector store?")
    existing_store = st.text_input("Existing vector store name (if updating):")
    new_store_name = st.text_input("New vector store name:")

    if st.button("Process Document"):
        embedding_storing(split_docs, create_new, existing_store, new_store_name)

# Initialize conversational model
if "conversation" not in st.session_state:
    token = os.getenv("HUGGINGFACE_TOKEN", "")
    vector_store_name = st.text_input("Vector store name to load:")
    if token and vector_store_name:
        st.session_state.conversation = prepare_rag_llm(
            token=token,
            vector_store_list=vector_store_name,
            temperature=0.7,
            max_length=256
        )

# Ask questions
question = st.text_input("Ask a question:")
if st.button("Get Answer"):
    if "conversation" in st.session_state:
        answer, sources = generate_answer(question, os.getenv("HUGGINGFACE_TOKEN", ""))
        st.write("### Answer:")
        st.write(answer)
        st.write("### Sources:")
        for source in sources:
            st.write(source)
    else:
        st.error("Model not initialized. Check token or vector store name.")
