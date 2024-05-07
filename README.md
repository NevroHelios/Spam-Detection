# Spam Detection Project

This project aims to detect spam emails using a machine learning model implemented in Python using the PyTorch library.

## Model Architecture

The machine learning model used in this project is `DetectSpamV0`, which is a simple feedforward neural network with the following architecture:


## Preprocessing

The text data for spam detection is preprocessed using NLTK (Natural Language Toolkit). The preprocessing steps include tokenization, removing punctuation, stemming, and removing stopwords.

## Directory Structure

- `mlapp.py`: The Flask application script.
- `model/`: Directory containing the trained machine learning model.
- `Preprocess/`: Directory containing the preprocessing code.
- `templates/`: Directory containing HTML templates for the web application.
- `requirements.txt`: File containing the Python dependencies required to run the project.

## Running the Project

To run the project, follow these steps:

1. Clone the repository: `git clone https://github.com/NevroHelios/Spam-Detection.git`
2. Navigate to the project directory: `cd spam-detection`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the Flask application: `python mlapp.py`
5. Open a web browser and go to `http://localhost:5000` to access the web application.

## Contact

For any questions or issues regarding this project, feel free to contact [me](mailto:dasharka05@gmail.com).
