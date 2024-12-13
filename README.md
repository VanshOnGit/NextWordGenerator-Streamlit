# Next Word Prediction App

This repository contains a Streamlit-based application for next-word prediction, developed under the guidance of Professor Nipun Batra at IIT Gandhinagar. The app is trained on *The Adventures of Sherlock Holmes* by Arthur Conan Doyle and provides meaningful predictions for user-input text.

## Features
- **Trained Dataset**: Utilizes text from *The Adventures of Sherlock Holmes* to generate predictions.
- **Handles Unknown Words**: Replaces unknown words with their closest known equivalent using word embeddings.
- **Embedding Support**: Works seamlessly with embedding dimensions of 128 and 256.
- **Word Relationship Visualization**: Interactive t-SNE plots to explore relationships like synonyms, antonyms, and more.
- **Device Adaptability**: Automatically utilizes GPU (if available) for faster computations.
- **User-Friendly Interface**: Streamlit-powered app for an interactive experience.

## Installation

1. Clone the repository:
   ```bash
   git clone <https://github.com/VanshOnGit/NextWordGenerator-Streamlit/>
   ```
2. Navigate to the project directory:
   ```bash
   cd <repository_directory>
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run streamlit_app.py
   ```

## Usage
1. Enter a text sequence in the input box.
2. Adjust parameters like embedding size, block size, and activation function from the sidebar.
3. Generate next-word predictions or visualize word relationships using the provided options.

## File Structure
- `streamlit_app.py`: Main Streamlit application file.
- `variables.pkl`: Pre-saved variables for the app (e.g., vocabulary, stoi/itos mappings).
- `models/`: Directory containing pre-trained model files.

## Requirements
- Python 3.8+
- Streamlit
- PyTorch
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn

Install all required packages using:
```bash
pip install -r requirements.txt
```

## Acknowledgements
This project was developed as part of a coursework assignment under the guidance of Professor Nipun Batra at IIT Gandhinagar.

