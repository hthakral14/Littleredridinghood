import streamlit as st
import torch
import os

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --------------------------------------------------
# Load LLM
# --------------------------------------------------
@st.cache_resource
def load_llm():
    model_id = "google/flan-t5-large"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # faster inference if GPU supports
        device_map="auto"
    )

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        temperature=0.0,
        do_sample=False
    )

    return HuggingFacePipeline(pipeline=pipe)

# --------------------------------------------------
# Load Story & Vector Store
# --------------------------------------------------
@st.cache_resource
def load_vector_store():
    if not os.path.exists("story.txt"):
        with open("story.txt", "w", encoding="utf-8") as f:
            f.write("Story file missing.")

    with open("story.txt", "r", encoding="utf-8") as f:
        story_text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=450,
        chunk_overlap=80
    )

    chunks = splitter.split_text(story_text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_texts(chunks, embeddings)

# --------------------------------------------------
# Initialize
# --------------------------------------------------
llm = load_llm()
vector_db = load_vector_store()

# --------------------------------------------------
# Question Type Classifier
# --------------------------------------------------
def classify_question(question: str) -> str:
    q = question.lower()

    if q.startswith("who is") or q.startswith("who was"):
        return "identity"

    if "evil" in q or "bad character" in q or "villain" in q:
        return "judgement"

    return "fact"

# --------------------------------------------------
# Pre-initialize QA chains for speed
# --------------------------------------------------
identity_template = """
Using ONLY the Story Context, describe who the character is
based on how they are introduced or described in the story.
If not described, say exactly: "Not mentioned in the story."

Story Context:
{context}

Question:
{question}

Answer:
"""

judgement_template = """
Using ONLY the Story Context, identify the character who causes
harm or deception through their actions.
Explain why based on story events.
If no such character exists, say exactly:
"Not mentioned in the story."

Story Context:
{context}

Question:
{question}

Answer:
"""

fact_template = """
Answer the question using ONLY the Story Context.
If the answer is not explicitly mentioned, say exactly:
"Not mentioned in the story."

Story Context:
{context}

Question:
{question}

Answer:
"""

identity_prompt = PromptTemplate(template=identity_template, input_variables=["context","question"])
judgement_prompt = PromptTemplate(template=judgement_template, input_variables=["context","question"])
fact_prompt = PromptTemplate(template=fact_template, input_variables=["context","question"])

qa_chains = {
    "identity": RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_type="mmr", search_kwargs={"k":4, "fetch_k":12}),
        chain_type_kwargs={"prompt": identity_prompt}
    ),
    "judgement": RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_type="mmr", search_kwargs={"k":6, "fetch_k":15}),
        chain_type_kwargs={"prompt": judgement_prompt}
    ),
    "fact": RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_type="mmr", search_kwargs={"k":6, "fetch_k":15}),
        chain_type_kwargs={"prompt": fact_prompt}
    )
}

# --------------------------------------------------
# Ask question
# --------------------------------------------------
def ask_question(question):
    q_type = classify_question(question)
    qa_chain = qa_chains.get(q_type, qa_chains["fact"])
    response = qa_chain.invoke({"query": question})
    return response["result"].strip()

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="AI chatbot")
st.title("Little Red Riding Hood : Chatbot")

st.markdown("""
**Tips for best results:**
- I know it might take time to show result, but please wait for the best answer
""")

# Initialize chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Keep only last 4 messages in memory
st.session_state.messages = st.session_state.messages[-4:]

# Display chat messages with prefixes
for msg in st.session_state.messages:
    if msg["role"] == "user":
        display_text = f"**User:** {msg['content']}"
    else:
        display_text = f"**Bot:** {msg['content']}"

    with st.chat_message(msg["role"]):
        st.markdown(display_text)

# User input
if user_input := st.chat_input("Ask a question about the story..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(f"**User:** {user_input}")

    # Generate bot response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            answer = ask_question(user_input)
            st.markdown(f"**Bot:** {answer}")
            
            # Add assistant message
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
    
    # Trim to last 4 messages
    st.session_state.messages = st.session_state.messages[-4:]
