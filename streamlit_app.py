

import streamlit as st
import torch
import pickle
from torch import nn
import os
import re
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import wordnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

# Load variables
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
    activation_fn = {"Tanh": torch.tanh, "ReLU": torch.relu, "Sigmoid": torch.sigmoid}[activation]
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

    # Handle unknown words in the context
    context = handle_unknown_words(context, model)

    # Convert context to indices and pad/truncate to fit block size
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

def plot_word_relations(model, num_words_per_category=10):
    embeddings = model.emb.weight.detach().cpu().numpy()

    categories = {'synonyms': [], 'antonyms': [], 'pronouns': [], 'names': [], 'unrelated': []}

    for word, idx in stoi.items():
        synsets = wordnet.synsets(word)
        if word in ['he', 'she', 'they', 'it']:
            categories['pronouns'].append(word)
        elif word.istitle():
            categories['names'].append(word)
        else:
            for syn in synsets:
                for lemma in syn.lemmas():
                    if lemma.name() in stoi and lemma.name() != word:
                        categories['synonyms'].append((word, lemma.name()))
                    if lemma.antonyms():
                        antonym = lemma.antonyms()[0].name()
                        if antonym in stoi:
                            categories['antonyms'].append((word, antonym))

    for category in categories:
        categories[category] = categories[category][:num_words_per_category]

    all_words, all_categories = [], []
    for category, word_pairs in categories.items():
        for word in word_pairs:
            if isinstance(word, tuple):
                all_words.extend(word)
                all_categories.extend([category] * 2)
            else:
                all_words.append(word)
                all_categories.append(category)

    word_indices = [stoi[word] for word in all_words if word in stoi]
    selected_embeddings = embeddings[word_indices]

    _, unique_indices = np.unique(selected_embeddings, axis=0, return_index=True)
    selected_embeddings = selected_embeddings[unique_indices]
    all_words = [all_words[i] for i in unique_indices]
    all_categories = [all_categories[i] for i in unique_indices]

    selected_embeddings += np.random.normal(0, 1e-3, selected_embeddings.shape)
    selected_embeddings /= np.linalg.norm(selected_embeddings, axis=1, keepdims=True)

    n_samples = len(selected_embeddings)
    perplexity = min(30, n_samples - 1)

    if n_samples < 2:
        st.error("Not enough unique points to generate a plot.")
        return

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
    tsne_embeddings = tsne.fit_transform(selected_embeddings)

    df = pd.DataFrame({'word': all_words, 'x': tsne_embeddings[:, 0], 'y': tsne_embeddings[:, 1], 'category': all_categories})

    plt.figure(figsize=(min(20, n_samples // 5), min(12, n_samples // 10)))
    sns.scatterplot(data=df, x='x', y='y', hue='category', style='category', s=100, palette='deep')

    for i, row in df.iterrows():
        plt.text(row['x'] + 0.01, row['y'] + 0.01, row['word'], fontsize=9)

    plt.title("Word Embeddings Visualization by Category")
    plt.grid(True)

    st.pyplot(plt)

st.title("Next Word Prediction App")
emb_dim = st.sidebar.selectbox("Embedding Dimension", [128, 256], index=1)
hidden_size = st.sidebar.selectbox("Hidden Size", [512, 1024], index=1)
activation = st.sidebar.selectbox("Activation Function", ["ReLU", "Tanh"])
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

num_words_per_category = st.sidebar.slider("Words per Category", min_value=10, max_value=100, value=50, step=10)

if st.sidebar.button("Visualize Word Relations"):
    st.write("Generating visualization...")
    plot_word_relations(model, num_words_per_category)




