Natural Language Processing (NLP) is a field concerned with the interaction between computers and human language, enabling computers to understand, interpret, and generate human language. 
Core NLP Concepts and Basic Code Examples:
Text Pre-processing: This involves cleaning and preparing text data for analysis.
Case Normalization: Converting all text to a consistent case (e.g., lowercase).
Python

        text = "Hello World"
        lower_case_text = text.lower()
        print(lower_case_text) # Output: hello world
Tokenization: Breaking text into individual words or sub-word units (tokens).
Python

        import nltk
        from nltk.tokenize import word_tokenize

        text = "Natural Language Processing is fascinating."
        tokens = word_tokenize(text)
        print(tokens) # Output: ['Natural', 'Language', 'Processing', 'is', 'fascinating', '.']
Stop Word Removal: Eliminating common words (e.g., "the", "a", "is") that often carry little meaning.
Python

        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        text = "This is a sample sentence with some stop words."
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
        print(filtered_sentence) # Output: ['sample', 'sentence', 'stop', 'words', '.']
Stemming/Lemmatization: Reducing words to their base or root form (e.g., "running" to "run").
Python

        from nltk.stem import PorterStemmer
        from nltk.stem import WordNetLemmatizer

        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        print(stemmer.stem("running")) # Output: run
        print(lemmatizer.lemmatize("running", pos='v')) # Output: run (requires 'wordnet' corpus)
Feature Extraction: Representing text data numerically for machine learning models.
TF-IDF (Term Frequency-Inverse Document Frequency): Weighing word importance based on frequency within a document and across a corpus.
Python

        from sklearn.feature_extraction.text import TfidfVectorizer

        documents = ["This is the first document.", "This document is the second document."]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        print(tfidf_matrix.shape) # Output: (2, 6) - 2 documents, 6 unique words