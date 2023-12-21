# Description: This file contains the code to train the model.
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, LSTM 
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.losses import BinaryCrossentropy
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer 
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import auc

# Load and preprocess dataset
phish_data = pd.read_csv('phishing_site_urls.csv')

# Tokenize and stem the URLs
tokenizer_regexp = RegexpTokenizer(r'[A-Za-z]+')
stemmer = SnowballStemmer("english")

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    # This function prints and plots the confusion matrix.
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Function to plot ROC curve
def plot_roc_curve(fpr, tpr, auc):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Function to tokenize and stem the text
def tokenize_and_stem(text):
    tokens = tokenizer_regexp.tokenize(text)
    return ' '.join(stemmer.stem(token) for token in tokens)

# Create 'text_sent' column with tokenized and stemmed text
phish_data['text_sent'] = phish_data['URL'].apply(tokenize_and_stem)

# Now you can use 'text_sent' with the Keras Tokenizer
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(phish_data['text_sent'])
sequences = tokenizer.texts_to_sequences(phish_data['text_sent'])

# Pad sequences to ensure uniform length
max_sequence_length = 300
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Create labels
labels = phish_data['Label'].map({'bad': 1, 'good': 0}).values

# Create a train/test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Load GloVe embeddings
embeddings_index = {}
with open('glove.6B/glove.6B.100d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Create embedding matrix
embedding_dim = 100
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Model configuration
additional_metrics = ['accuracy']
batch_size = 128
embedding_output_dims = 15
# loss_function = BinaryCrossentropy
loss_function = MeanSquaredError()
number_of_epochs = 5
validation_split = 0.20
verbosity_mode = 1

# # Define the Keras model
# model = Sequential()
# model.add(Embedding(5000, embedding_output_dims, input_length=max_sequence_length))
# model.add(LSTM(10))
# model.add(Dense(1, activation='sigmoid'))

# Model with GloVe Embedding Layer
model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1,
                    embedding_dim,
                    weights=[embedding_matrix],
                    input_length=max_sequence_length,
                    trainable=False))
model.add(LSTM(10))
# model.add(LSTM(50))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01)))

# # Compile the model
# model.compile(optimizer='adam', loss=loss_function, metrics=additional_metrics)

# Compile the model
model.compile(optimizer='SGD', loss=loss_function, metrics=additional_metrics)

# Train the model
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=number_of_epochs, verbose=verbosity_mode, validation_split=validation_split)

# Evaluate the model after training
test_results = model.evaluate(X_test, y_test, verbose=1)
predicted_probs = model.predict(X_test)
roc = roc_auc_score(y_test, predicted_probs)
print("AUC-ROC:", roc)

# Plotting the loss and accuracy
plt.figure(figsize=(14, 5))

# Plotting the loss plot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

# Plotting the ROC curve
fpr, tpr, _ = roc_curve(y_test, predicted_probs)
roc_auc = auc(fpr, tpr)

plt.subplot(1, 2, 2)
plot_roc_curve(fpr, tpr, roc_auc)

plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, predicted_probs.round())
print("Confusion Matrix:")
print(cm)

# Plot Confusion Matrix
plot_confusion_matrix(cm, classes=['Good', 'Bad'])

# Classification Report
print("Classification Report:")
print(classification_report(y_test, predicted_probs.round()))

# Save the trained model
model.save('phishing_detection_model_new4(base).h5')