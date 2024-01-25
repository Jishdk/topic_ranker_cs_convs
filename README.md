# topic_ranker_cs_convs
This project aims to perform topic modelling and classification on a text dataset using Latent Dirichlet Allocation (LDA) and Logistic Regression.


README
Project Description

This project aims to perform topic modelling and classification on a text dataset using Latent Dirichlet Allocation (LDA) and Logistic Regression. The text data is pre-processed, and an LDA model is trained to extract topics. The dataset is then labelled with the most dominant topic, and a Logistic Regression classifier is trained for topic classification. The trained LDA model, classifier, and related objects are saved to files for later use.
Installation Instructions

    Install required Python libraries:
    pip install pandas nltk gensim scikit-learn matplotlib seaborn pickle
    Download the code and accompanying files from the repository.
    Run the Python script in your preferred environment.

Example / Tutorial

    Load the dataset from the specified URL and store it in a pandas Data Frame.
    Perform EDA and visualization on the dataset to understand the text length distribution.
    Pre-process the text by lowercasing, removing URLs, punctuations, tokenizing, filtering out stopwords and short words, and lemmatizing.
    Train an LDA model on the pre-processed text to extract topics.
    Assign the most dominant topic to each document in the dataset.
    Train a Logistic Regression classifier on the labelled dataset for topic classification.
    Evaluate the classifier's performance using accuracy and F1-score metrics.
    Use the TopicPredictor class to predict the topic of a given conversation.

API Documentation
preprocess_text(text: str) -> List[str]

    Function to preprocess a given text.
    Parameters:
        text: A string containing the text to pre-process.
    Returns: A list of pre-processed tokens.

get_dominant_topic(ldamodel: LdaModel, corpus: List, texts: List[str]) -> pd.DataFrame

    Function to get the most dominant topic for each document in the dataset.
    Parameters:
        ldamodel: A trained LdaModel object.
        corpus: A list representing the corpus for the LDA model.
        texts: A list of strings containing the original text data.
    Returns: A DataFrame containing the dominant topic, percent contribution, and topic keywords for each document.

TopicPredictor Class

    A class that loads the trained LDA model, classifier, preprocessor, and other objects and provides a method to predict the topic of a given conversation.
    Methods:
        predict_topic(conversation: str) -> Tuple[int, str]: Predicts the topic ID and topic name for a given conversation.
        get_topic_name(topic_id: int) -> str: Gets the topic name given a topic ID.

Usage Example

    Create an instance of the TopicPredictor class.
    Run an interactive loop to predict topics for user input.

topic_predictor = TopicPredictor()
conversation = input("Enter a conversation (type 'exit' to quit): ")
dominant_topic = topic_predictor.predict_topic(conversation)
topic_name = topic_predictor.get_topic_name(dominant_topic)
print(f"Predicted topic ID: {dominant_topic}")
print(f"Predicted topic: {topic_name}\n")

Running the Django Server

Run the following command in the terminal to start the Django development server:

python manage.py runserver

Access the web application at http://127.0.0.1:8000/ in your browser.
