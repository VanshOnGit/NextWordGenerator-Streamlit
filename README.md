# Next Word Prediction App

Deployed App: https://vansh-textgen.streamlit.app/

This repository contains a Streamlit-based application for next-word prediction, developed under the guidance of Professor Nipun Batra at IIT Gandhinagar. The app is trained on *The Adventures of Sherlock Holmes* by Arthur Conan Doyle and provides predictions for user-input text.

## Features
- **Trained Dataset**: Utilizes text from *The Adventures of Sherlock Holmes* to generate predictions.
- **Handles Unknown Words**: Replaces unknown words with their closest known equivalent using word embeddings.
- **Embedding Support**: Works seamlessly with embedding dimensions of 128 and 256.
- **Word Relationship Visualization**: Interactive t-SNE plots to explore relationships like synonyms, antonyms, and more.
- **Device Adaptability**: Automatically utilizes GPU (if available) for faster computations.
- **User-Friendly Interface**: Streamlit-powered app for an interactive experience.

## Usage
1. Enter a text sequence in the input box.
2. Adjust parameters like embedding size, block size, and activation function from the sidebar.
3. Generate next-word predictions.

