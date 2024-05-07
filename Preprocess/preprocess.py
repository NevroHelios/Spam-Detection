import nltk
import string
import pickle
import torch
from nltk.tokenize import word_tokenize

try:
    # Try importing PorterStemmer and stopwords
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords
except:
    # If import fails, download necessary NLTK resources
    nltk.download('punkt')
    nltk.download('stopwords')
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords

# path to the saved vectorizer and load it
VEC_PATH = "Preprocess/tfidf_vectorizer.pkl"
with open(VEC_PATH, 'rb') as file:
    tfidf = pickle.load(file)

def pre_process(text):
    # Lowercasing
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Stopword Removal
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Punctuation Removal
    tokens = [word for word in tokens if word not in string.punctuation]

    # Stemming
    ps = PorterStemmer()
    stemmed_tokens = [ps.stem(word) for word in tokens]

    # Join tokens back into a string
    processed_text = ' '.join(stemmed_tokens)

    # return processed_text as a torch.Tensor
    processed_text = tfidf.transform([processed_text]).toarray().astype('float32')
    processed_text = torch.from_numpy(processed_text)
    
    return processed_text