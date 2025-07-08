import streamlit as st
import torch
import pickle
from torch import nn
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import base64


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")


with open('variables.pkl', 'rb') as f:
    variables = pickle.load(f)

vocab_size = variables['vocab_size']
stoi = variables['stoi']
itos = variables['itos']
default_block_size = variables['block_size']
default_random_seed = variables['random_seed']

class NextWordPredictor(nn.Module):
    def __init__(self, block_size, vocab_size, emb_dim, hidden_size, activation):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
        self.activation = activation
        self.lin2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.emb(x).view(x.size(0), -1)
        x = self.activation(self.lin1(x))
        return self.lin2(x)

@st.cache_resource
def load_model(emb_dim, hidden_size, activation, block_size, random_seed):
    activation_fn = {"tanh": torch.tanh, "relu": torch.relu, "sigmoid": torch.sigmoid}[activation]
    model_path = os.path.join("models", f"model_{emb_dim}_{hidden_size}_{activation}_bs{block_size}_rs{random_seed}.pt")

    if not os.path.exists(model_path):
        st.error(f"Model {model_path} not found. Make sure the model file exists.")
        st.stop()

    model = NextWordPredictor(block_size, vocab_size, emb_dim, hidden_size, activation_fn).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

nltk.download('wordnet')
nltk.download('omw-1.4')

def find_closest_word(embedding, model):
    word_embeddings = model.emb.weight.detach().cpu().numpy()
    similarities = cosine_similarity(embedding.reshape(1, -1), word_embeddings)
    closest_idx = np.argmax(similarities)
    return itos.get(closest_idx, '<UNK>')

def handle_unknown_words(words, model):
    processed_words = []
    word_embeddings = model.emb.weight.detach().cpu().numpy()

    for word in words:
        if word in stoi:
            processed_words.append(word)
        else:
            random_vector = np.random.randn(word_embeddings.shape[1])
            closest_word = find_closest_word(random_vector, model)
            st.warning(f"Unknown word '{word}' replaced with '{closest_word}'")
            processed_words.append(closest_word)

    return processed_words

def generate_text(model, context, max_words, block_size):
    model.eval()

    context = handle_unknown_words(context, model)

    context_indices = [stoi.get(word, stoi['<PAD>']) for word in context]
    context_indices = [0] * (block_size - len(context_indices)) + context_indices[-block_size:]

    generated_text = []
    for _ in range(max_words):
        x = torch.tensor([context_indices], dtype=torch.long).to(device)

        with torch.no_grad():
            logits = model(x)

        probs = F.softmax(logits, dim=-1).squeeze(0)
        next_word_idx = torch.multinomial(probs, 1).item()
        next_word = itos.get(next_word_idx, '<UNK>')

        if next_word == '<PAD>':
            break

        generated_text.append(next_word)
        context_indices = context_indices[1:] + [next_word_idx]

    return ' '.join(generated_text) if generated_text else '<No valid output>'

import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

image_base64 = get_base64_image("logo.png")  

st.markdown(
    f"""
    <div style='text-align: center;'>
        <img src='data:image/png;base64,{image_base64}' width='400'/>
    </div>
    """,
    unsafe_allow_html=True
)


st.title("SherlockNext - Next Word Prediction App")
emb_dim = st.sidebar.selectbox("Embedding Dimension", [128, 256], index=1)
hidden_size = st.sidebar.selectbox("Hidden Size", [512, 1024], index=1)
activation = st.sidebar.selectbox("Activation Function", ["relu", "tanh"])

block_size = st.sidebar.slider("Block Size", 3, 15, default_block_size)
random_seed = st.sidebar.number_input("Random Seed", min_value=0, value=default_random_seed)

model = load_model(emb_dim, hidden_size, activation, block_size, random_seed)

user_input = st.text_area("Enter some text to generate the next words:")
if user_input:
    context = user_input.lower().split()
    max_words = st.slider("Number of Words to Predict", 1, 200, 20)
    with st.spinner("Generating text..."):
        generated_text = generate_text(model, context, max_words, block_size)
    st.write(f"**Generated Text:** {generated_text}")
