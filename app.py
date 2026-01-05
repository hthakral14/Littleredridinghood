import streamlit as st
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
from collections import deque # Required for limited memory
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch


# Load Models
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
    model_name = "google/flan-t5-base" 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
    return embed_model, generator

embed_model, llm = load_models()


# Load Story
@st.cache_data
def load_story():
    try:
        with open("story.txt", "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        text = "Little Red Riding Hood went to her grandmother's house. She met a wolf in the woods."

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    embeddings = embed_model.encode(sentences, convert_to_tensor=True)
    return sentences, embeddings

sentences, sentence_embeddings = load_story()


# Logic
def ask_question(question):
    q_emb = embed_model.encode(question, convert_to_tensor=True)
    hits = util.cos_sim(q_emb, sentence_embeddings)[0]
    top_results = torch.topk(hits, k=min(3, len(sentences)))
    
    if top_results.values[0] < 0.40:
        return "Don't know"

    context_text = " ".join([sentences[i] for i in top_results.indices])
    prompt = (
        f"Answer the following question using only the provided context. "
        f"If the answer is not in the context, say 'Don't know'.\n\n"
        f"Context: {context_text}\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    result = llm(prompt, max_new_tokens=50, do_sample=False, temperature=0.0)[0]["generated_text"].strip()
    return result if result else "Don't know"

# Streamlit UI with Limited Memory
st.title("Little Red Riding Hood AI")
st.write("Ask anything based on the story. Unknown questions return *Don't know*.")


# Initialize limited memory (maxlen=3 keeps only last 3 exchanges)
if "chat_memory" not in st.session_state or isinstance(st.session_state.chat_memory, list):
    st.session_state.chat_memory = deque(maxlen=3)

# Display chat history with User: and Bot:
for chat in st.session_state.chat_memory:
    st.markdown(f"**User:** {chat['q']}")
    st.markdown(f"**Bot:** {chat['a']}")
    st.markdown("---")

# Input field
user_input = st.chat_input("Ask about the story...")

if user_input:
    answer = ask_question(user_input)
    
    # Add to deque; oldest item is removed automatically if size > 3
    st.session_state.chat_memory.append({"q": user_input, "a": answer})
    
    # Rerun to show the updated labels immediately
    st.rerun()