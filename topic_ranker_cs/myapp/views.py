# Import required modules
from django.shortcuts import render
from .prediction_interface import TopicPredictor

# Initialize the TopicPredictor instance
topic_predictor = TopicPredictor()

# Define the home view
def home(request):
    # Check if the request method is POST (i.e., the form was submitted)
    if request.method == 'POST':
        # Extract the conversation text from the form submission
        conversation = request.POST.get('conversation')
        # Get the predicted topic ID and topic name using the TopicPredictor instance
        dominant_topic_id, dominant_topic_name = topic_predictor.predict_topic(conversation)
        # Render the results page with the topic ID and topic name as context data
        return render(request, 'results.html', {'topic_id': dominant_topic_id, 'topic_name': dominant_topic_name})
    else:
        # Render the home page if the request method is not POST
        return render(request, 'home.html')

# Define the results view
def results(request, topic_id, topic_name):
    # Render the results page with the provided topic ID and topic name as context data
    return render(request, 'results.html', {'topic_id': topic_id, 'topic_name': topic_name})
