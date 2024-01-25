# Import required modules
import pickle
from gensim.models import LdaModel
from gensim import corpora

# Define the TopicPredictor class
class TopicPredictor:
    def __init__(self):
        # Load the LDA model and dictionary
        self.lda_model = LdaModel.load('lda_model.pkl')
        self.id2word = corpora.Dictionary.load('dictionary.pkl')

        # Load the preprocessor, classifier, and vectorizer from pickled files
        with open('preprocessor.pkl', 'rb') as f:
            self.preprocess_fn = pickle.load(f)
        with open('classifier.pkl', 'rb') as f:
            self.classifier = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)

    def predict_topic(self, conversation):
        # Preprocess the conversation text
        preprocessed_text = self.preprocess_fn(conversation)
        # Convert preprocessed text to a bag-of-words representation
        text_bow = self.id2word.doc2bow(preprocessed_text)
        # Get the topic scores for the conversation
        topic_scores = self.lda_model.get_document_topics(text_bow)
        # Find the dominant topic
        dominant_topic = max(topic_scores, key=lambda x: x[1])
        # Extract the topic ID and topic name
        topic_id = dominant_topic[0]
        topic_name = self.get_topic_name(topic_id)
        # Return the topic ID and topic name
        return topic_id, topic_name

    def get_topic_name(self, topic_id):
        # Get the topic name given a topic ID
        return self.lda_model.print_topic(topic_id)

# Define the main function
def main():
    # Create an instance of TopicPredictor
    topic_predictor = TopicPredictor()

    # Run an interactive loop to predict topics for user input
    while True:
        conversation = input("Enter a conversation (type 'exit' to quit): ")
        if conversation.strip().lower() == 'exit':
            break
        else:
            dominant_topic = topic_predictor.predict_topic(conversation)
            topic_name = topic_predictor.get_topic_name(dominant_topic)
            print(f"Predicted topic ID: {dominant_topic}")
            print(f"Predicted topic: {topic_name}\n")

# Run the main function if this script is executed
if __name__ == '__main__':
    main()