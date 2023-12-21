import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer 
from nltk.stem.snowball import SnowballStemmer
import tensorflow as tf

# Load the dataset
phish_data = pd.read_csv('phishing_site_urls.csv')

# Initialize tokenizer and stemmer
tokenizer_keras = Tokenizer(num_words=5000)
tokenizer_regexp = RegexpTokenizer(r'[A-Za-z]+')
stemmer = SnowballStemmer("english")

# Function to tokenize and stem the text
def tokenize_and_stem(text, tokenizer_regexp, stemmer):
    # Tokenize with NLTK RegexpTokenizer
    tokens = tokenizer_regexp.tokenize(text)
    # Stem with NLTK SnowballStemmer
    stems = [stemmer.stem(token) for token in tokens]
    return ' '.join(stems)

# Apply the function to each URL and create the 'text_sent' column
phish_data['text_sent'] = phish_data['URL'].apply(lambda x: tokenize_and_stem(x, tokenizer_regexp, stemmer))

# Fit the Keras Tokenizer
tokenizer_keras.fit_on_texts(phish_data['text_sent'])

# Define the max_sequence_length based on your analysis of the data
max_sequence_length = 300

# Function to preprocess URLs
def preprocess_urls(url_list, tokenizer_keras, max_sequence_length, tokenizer_regexp, stemmer):
    preprocessed_urls = [tokenize_and_stem(url, tokenizer_regexp, stemmer) for url in url_list]
    sequences = tokenizer_keras.texts_to_sequences(preprocessed_urls)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequences

# Function to predict labels for new URLs
def predict_urls(url_list, model, tokenizer_keras, max_sequence_length, tokenizer_regexp, stemmer):
    padded_sequences = preprocess_urls(url_list, tokenizer_keras, max_sequence_length, tokenizer_regexp, stemmer)
    predictions = model.predict(padded_sequences, verbose=1)
    predicted_labels = [1 if prediction > 0.5 else 0 for prediction in predictions]
    return predicted_labels

# Example usage:
url_list = [
    'www.google.com/',
    'finviz.com/map.ashx?t=sec',
    'nobell.it/70ffb52d079109dca5664cce6f317373782/login.SkyPe.com/en/cgi-bin/verification/login/70ffb52d079109dca5664cce6f317373/index.php?cmd=_profile-ach&outdated_page_tmpl=p/gen/failed-to-load&nav=0.5.1&login_access=1322408526',
    'www.dghjdgf.com/paypal.co.uk/cycgi-bin/webscrcmd=_home-customer&nav=1/loading.php',
    'macaulay.cuny.edu/CUFF/?q=node/61',
    'macaulay.cuny.edu/eportfolios/jgriffith/beautiful-creatures-a-green-opera/',
    'macbsp.com/english.aspx',
    'macdonaldcampusathletics.mcgill.ca/',
    'macdonaldcampusathletics.mcgill.ca/webpages/awards.htm',
    'macdonrod.blogspot.com/',
    'macdougal.com/'
]

# Load the trained model
model = tf.keras.models.load_model('phishing_detection_model.h5')

# Predict labels for new URLs
predicted_labels = predict_urls(url_list, model, tokenizer_keras, max_sequence_length, tokenizer_regexp, stemmer)

# Display predictions
for url, label in zip(url_list, predicted_labels):
    print(f'URL: {url}, Predicted Label: {"Phishing" if label == 1 else "Not Phishing"}')