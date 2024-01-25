# Import necessary libraries
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import gensim
from gensim import corpora
from gensim.models import LdaModel
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Record the start time for tracking processing time
start_time = time.time()

# Download required NLTK data for stopwords, lemmatization, and tokenization
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Load the dataset from the specified URL
url = "https://ciphix.io/ai/data.csv"
data = pd.read_csv(url, delimiter=',', names=['text']).dropna()

# Perform Exploratory Data Analysis (EDA) with visualization

# Print the first 5 rows and last 5 rows of the dataset
print("First 5 rows:\n", data.head())
print("\nLast 5 rows:\n", data.tail())

# Print the shape, column names, and data types of the dataset
print("Shape of the dataset:", data.shape)
print("\nColumn names:", data.columns)
print("\nData types:\n", data.dtypes)

# Use a smaller subset of the dataset (e.g., first 10,000 rows) for faster processing
data = data.head(10000)

# Add a new column to the dataset with the length of each text
data['text_length'] = data['text'].apply(len)

# Plot a histogram of the text lengths
plt.figure(figsize=(10, 5))
sns.histplot(data=data, x='text_length', bins=50)
plt.xlabel("Text Length")
plt.ylabel("Frequency")
plt.title("Distribution of Text Lengths")
plt.show()

# Drop the 'text_length' column after visualization
data.drop('text_length', axis=1, inplace=True)

# Preprocess the text

# Define stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Group the data by topic label and count the number of occurrences
topic_counts = data.groupby('text').size().reset_index(name='count').sort_values('count', ascending=False)

# Print the top 10 topics with their corresponding counts
print("Top 10 Topics:\n")
for idx, row in topic_counts.head(10).iterrows():
    print(f"Topic {row['text']}: {row['count']} occurrences", "\n")

# Preprocess the text by separating the words and applying transformations
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)

    # Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    # Remove stopwords and short words
    words = [word for word in words if word not in stop_words and len(word) > 2]

    # Lemmatize the words
    words = [lemmatizer.lemmatize(word) for word in words]

    return words

# Apply preprocessing to the text column and store the result in a new column
data['preprocessed_text'] = data['text'].apply(preprocess_text)

# Prepare the data for LDA modeling by tokenizing the preprocessed text and creating a dictionary and corpus
texts = data['preprocessed_text'].tolist()
id2word = corpora.Dictionary(texts)
id2word.filter_extremes(no_below=20, no_above=0.5)
corpus = [id2word.doc2bow(text) for text in texts]

# Train a Word2Vec model on the preprocessed text for the topic names
w2v_model = gensim.models.Word2Vec(texts, vector_size=100, window=5, min_count=2, workers=4)

#Generate the topic name
def generate_topic_name(w2v_model, keywords):
    most_similar_word = w2v_model.wv.most_similar(positive=keywords, topn=1)[0][0]
    return most_similar_word

# Set the number of topics for LDA
num_topics = 10

# Train the LDA model with 10 passes
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=num_topics,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)

# Save the trained LDA model to a file
pickle.dump(lda_model, open("lda_model.pkl", "wb"))

# Save the dictionary to a file
id2word.save('dictionary.pkl')

# Save the text preprocessor function to a file
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocess_text, f)

# Display the top keywords for each topic in the LDA model
topics = lda_model.show_topics(num_topics=num_topics, num_words=10, formatted=False)
topic_labels = {}  # Store topic labels
for idx, topic in topics:
    print(f"Topic: {idx + 1}")
    keywords = [word for word, _ in topic]
    print("Keywords: ", ", ".join(keywords))

    # Generate a topic name based on the most similar word to the top keywords
    topic_name = generate_topic_name(w2v_model, keywords)
    topic_labels[idx] = topic_name
    print(f"Generated topic name: {topic_name}\n")


# Assign topic labels to the dataset based on the LDA model's results by finding the most dominant topic
def get_dominant_topic(ldamodel, corpus, texts):
    # Initialize the dataframe
    topic_info = []

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = row[0] if ldamodel.per_word_topics else row

        # Check if the row is not empty
        if row:
            row = sorted(row, key=lambda x: (x[1]), reverse=True)

            # Get the dominant topic, percent contribution, and keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    topic_info.append([int(topic_num), round(prop_topic, 4), topic_keywords])
                else:
                    break
        else:
            topic_info.append([None, None, None])

    df_topic_sents_keywords = pd.DataFrame(topic_info,
                                           columns=['Dominant_Topic', 'Percent_Contribution', 'Topic_Keywords'])

    # Add the original text to the end of the output
    contents = pd.Series(texts)
    df_topic_sents_keywords = pd.concat([df_topic_sents_keywords, contents], axis=1)
    return df_topic_sents_keywords

df_topic_sents_keywords = get_dominant_topic(lda_model, corpus, data['text'])

# Add the assigned topic labels to the dataset
data['topic_label'] = df_topic_sents_keywords['Dominant_Topic'].apply(lambda x: topic_labels.get(x))

# Group the data by topic label and count the number of occurrences
labeled_topic_counts = data.groupby('topic_label').size().reset_index(name='count').sort_values('count', ascending=False)

# Print the top 10 labeled topics with their corresponding counts
print("Top 10 Labeled Topics:\n")
for idx, row in labeled_topic_counts.head(10).iterrows():
    print(f"{row['topic_label']}: {row['count']} occurrences", "\n")

# Train the classifier using preprocessed text and assigned topic labels
X = data['preprocessed_text']
y = data['topic_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate a TfidfVectorizer and transform the preprocessed text into a matrix of TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train.apply(lambda x: ' '.join(x)))
X_test_tfidf = vectorizer.transform(X_test.apply(lambda x: ' '.join(x)))

# Train a Logistic Regression classifier with the TF-IDF feature matrix and the assigned topic labels
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_tfidf, y_train)

# Save the trained classifier and vectorizer to files
with open('classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Evaluate the classifier on the test set
y_pred = classifier.predict(X_test_tfidf)

# Calculate and print the evaluation metrics for the classifier
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"Accuracy: {accuracy:.2f}")
print(f"F1-score: {f1:.2f}")


# Record the end time
end_time = time.time()

# Calculate the elapsed time and print it out
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")