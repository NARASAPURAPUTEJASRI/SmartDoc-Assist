Hugging Face Transformers for sentiment analysis

from transformers import pipeline
import torch
# Load a pre-trained sentiment analysis model
# 'sentiment-analysis' pipeline defaults to 'distilbert-base-uncased-finetuned-sst-2-english'
sentiment_pipeline = pipeline("sentiment-analysis")

text1 = "This movie was absolutely fantastic!"
text2 = "The food was okay, but the service was terrible."
text3 = "I'm not sure how I feel about this."

results1 = sentiment_pipeline(text1)
results2 = sentiment_pipeline(text2)
results3 = sentiment_pipeline(text3)

print(f"'{text1}' - Sentiment: {results1[0]['label']} (Score: {results1[0]['score']:.4f})")
print(f"'{text2}' - Sentiment: {results2[0]['label']} (Score: {results2[0]['score']:.4f})")
print(f"'{text3}' - Sentiment: {results3[0]['label']} (Score: {results3[0]['score']:.4f})")


2. Named Entity Recognition (NER) with spaCy

import spacy

# Load the pre-trained English model (small version)
# If you haven't downloaded it, run: !python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

text = "Apple is looking at buying U.K. startup for $1 billion from London."

doc = nlp(text)

print("Named Entities:")
for ent in doc.ents:
    print(f"  {ent.text:<15} {ent.label_:<10} ({spacy.explain(ent.label_):<30})") # Explain entity labels 


3. Text summarization with Hugging Face Transformers

from transformers import pipeline

# Load a pre-trained summarization model
summarizer = pipeline("summarization")

text = """
Hugging Face is a company that builds tools and libraries for natural language processing (NLP).
Their transformers library is a popular open-source library that provides pre-trained models for various NLP tasks,
including text summarization. These models are based on the transformer architecture, which has shown remarkable
performance in many NLP benchmarks.  Text summarization is the process of creating a shorter version of a text document
that still conveys the main ideas. It can be broadly categorized into two types: extractive and abstractive.
Extractive summarization selects important sentences or phrases directly from the original text, while abstractive
summarization generates new sentences to capture the key information. Hugging Face's transformers library allows
developers to easily implement both types of summarization.
"""

summary = summarizer(text, max_length=50, min_length=20, do_sample=False) # Specify max and min lengths

print("Original Text:\n", text)
print("\nSummarized Text:\n", summary[0]['summary_text'])


4. Word embeddings (Word2Vec) with Gensim
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

# Ensure you have the 'punkt' tokenizer downloaded
nltk.download('punkt')

sentences = [
    "I like to play basketball",
    "He likes to play football",
    "She loves reading books",
    "My dog loves to run and jump",
    "Basketball is a fun sport",
]

# Tokenize sentences
tokenized_sentences = [word_tokenize(sent.lower()) for sent in sentences]

# Train the Word2Vec model
# vector_size: dimension of the word vectors
# window: context window size
# min_count: ignore words with frequency lower than this
model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, sg=0) # Using CBOW (sg=0)

# Access word vectors
print("Vector for 'basketball':", model.wv['basketball'])

# Find similar words
similar_words = model.wv.most_similar('basketball')
print("\nWords similar to 'basketball':", similar_words)

print("\nCosine similarity between 'basketball' and 'football':", model.wv.similarity('basketball', 'football'))
print("Cosine similarity between 'basketball' and 'books':", model.wv.similarity('basketball', 'books')) 


5. Recurrent Neural Networks (RNNs) for text classification (with Keras)
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Sample data
texts = [
    "This is a positive review.",
    "I loved this movie, it was great!",
    "The service was terrible and I hated it.",
    "Not a bad experience overall.",
    "What a wonderful day!",
    "This product is awful.",
    "Highly recommended, excellent quality.",
    "Very disappointing, a waste of money.",
]
labels = np.array([1, 1, 0, 1, 1, 0, 1, 0])  # 1 for positive, 0 for negative

# Tokenization
tokenizer = Tokenizer(num_words=100) # Max 100 words
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Padding sequences to a fixed length
maxlen = max(len(s) for s in sequences)
padded_sequences = pad_sequences(sequences, maxlen=maxlen)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)

# Build the RNN model
model = Sequential([
    Embedding(input_dim=100, output_dim=32, input_length=maxlen), # Embedding layer
    SimpleRNN(units=32), # Simple RNN layer
    Dense(units=1, activation='sigmoid') # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=2) # Train for 10 epochs

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Make predictions
new_text = ["This was a fantastic product, very happy."]
new_sequence = tokenizer.texts_to_sequences(new_text)
new_padded_sequence = pad_sequences(new_sequence, maxlen=maxlen)
prediction = model.predict(new_padded_sequence)
predicted_sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
print(f"'{new_text[0]}' - Predicted Sentiment: {predicted_sentiment} (Score: {prediction[0][0]:.4f})")
