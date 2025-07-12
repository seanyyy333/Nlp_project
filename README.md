# Nlp_project
# Assuming your provided installations (nltk, spacy, textblob) are done
# Further advanced setup would involve deep learning frameworks like TensorFlow or PyTorch.

# !pip install transformers requests beautifulsoup4 # For more advanced features

import nltk
import spacy
from textblob import TextBlob
import random
import requests
from bs4 import BeautifulSoup
from collections import deque # For more advanced context memory

# --- Advanced NLP Model (Conceptual - would require actual model loading/inference) ---
# In a real-world scenario, you'd load a pre-trained model like a smaller GPT variant
# or a custom-trained seq2seq model here.
# For demonstration, we'll simulate a slightly more intelligent response.

def advanced_nlp_response(user_input, chat_history_vectors):
    """
    Conceptual function to simulate an advanced NLP model's response.
    In reality, this would involve:
    1. Tokenizing user_input.
    2. Encoding user_input (and potentially chat_history_vectors if using an attention mechanism).
    3. Passing through a deep learning model (e.g., Transformer decoder).
    4. Decoding the model's output into a natural language response.
    """
    user_input_lower = user_input.lower()

    if "weather in" in user_input_lower:
        city = user_input_lower.split("weather in")[-1].strip().replace("?", "").replace(".", "")
        return get_weather(city)
    elif "news about" in user_input_lower:
        topic = user_input_lower.split("news about")[-1].strip().replace("?", "").replace(".", "")
        return get_news(topic)
    elif "tell me about" in user_input_lower:
        topic = user_input_lower.split("tell me about")[-1].strip().replace("?", "").replace(".", "")
        return get_wikipedia_summary(topic)
    elif "who are you" in user_input_lower:
        return "I am a helpful AI assistant, designed to process information and assist you."
    elif "what can you do" in user_input_lower:
        return "I can answer questions, provide information, analyze sentiment, and much more, depending on my integrations."

    # Fallback to a slightly more varied response for unknown inputs
    responses = [
        f"That's an interesting thought about '{user_input}'. Can you tell me more?",
        "I'm still learning, but I'm trying my best to understand.",
        "Could you elaborate on that?",
        "I'm not quite sure I follow, but I'm here to help if you rephrase.",
        "What are your thoughts on that?"
    ]
    return random.choice(responses)


# --- Sentiment Analysis (using TextBlob - as provided) ---
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# --- Contextual Memory (More robust with a deque for limited history) ---
# Using a deque for a fixed-size conversation history
conversation_history = deque(maxlen=5) # Stores last 5 turns (user + bot)

def add_to_history(speaker, text):
    conversation_history.append((speaker, text))

def get_context_summary():
    """
    Summarizes the recent conversation history.
    In an advanced system, this would be fed into the NLP model's attention.
    """
    context = []
    for speaker, text in conversation_history:
        context.append(f"{speaker}: {text}")
    return "\n".join(context)

# --- Real-time API Integration (Functional Placeholders) ---

# OpenWeatherMap API - Get your API key from openweathermap.org
WEATHER_API_KEY = "YOUR_OPENWEATHERMAP_API_KEY"
def get_weather(city):
    """Fetches current weather for a city using OpenWeatherMap API."""
    if not WEATHER_API_KEY or WEATHER_API_KEY == "YOUR_OPENWEATHERMAP_API_KEY":
        return f"Please set your OpenWeatherMap API key to get real weather for {city}."
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        data = response.json()
        if data["cod"] == 200:
            weather_desc = data['weather'][0]['description']
            temp = data['main']['temp']
            return f"The weather in {city} is {weather_desc} with a temperature of {temp}°C."
        else:
            return f"Could not get weather for {city}. Error: {data.get('message', 'Unknown error')}"
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather for {city}: {e}"
    except KeyError:
        return f"Could not parse weather data for {city}. Please check the city name."

# News API - Get your API key from newsapi.org
NEWS_API_KEY = "YOUR_NEWSAPI_API_KEY"
def get_news(topic):
    """Fetches top news headlines for a topic using News API."""
    if not NEWS_API_KEY or NEWS_API_KEY == "YOUR_NEWSAPI_API_KEY":
        return f"Please set your News API key to get real news about {topic}."
    try:
        url = f"https://newsapi.org/v2/everything?q={topic}&language=en&sortBy=relevancy&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        articles = data.get('articles', [])
        if articles:
            headlines = [f"{i+1}. {a['title']} ({a['source']['name']})" for i, a in enumerate(articles[:3])]
            return f"Here are some top headlines about {topic}:\n" + "\n".join(headlines)
        else:
            return f"No news found about {topic}."
    except requests.exceptions.RequestException as e:
        return f"Error fetching news about {topic}: {e}"

# Wikipedia API (Conceptual - using requests and BeautifulSoup for simplicity)
def get_wikipedia_summary(query):
    """Fetches a summary from Wikipedia."""
    try:
        search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={query}&limit=1&format=json"
        search_response = requests.get(search_url).json()
        page_title = search_response[1][0] if search_response[1] else None

        if page_title:
            summary_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts&exintro&explaintext&redirects=1&titles={page_title}&format=json"
            summary_response = requests.get(summary_url).json()
            pages = summary_response['query']['pages']
            page_id = next(iter(pages))
            summary = pages[page_id].get('extract', 'No summary found.')
            return f"According to Wikipedia about {page_title}: {summary[:300]}..." # Truncate for brevity
        else:
            return f"Could not find information about '{query}' on Wikipedia."
    except requests.exceptions.RequestException as e:
        return f"Error fetching Wikipedia data: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# --- Main Interaction Loop with Enhancements ---

print("Advanced Chatbot: Hello! How can I help you today? (Type 'quit' or 'exit' to end)")
print("Advanced Chatbot: Try asking about 'weather in London' or 'news about AI' or 'tell me about quantum physics'.")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("Advanced Chatbot: Goodbye! It was a pleasure assisting you.")
        break

    add_to_history("User", user_input)

    # Analyze sentiment
    sentiment = analyze_sentiment(user_input)
    if sentiment > 0.3:
        print("Advanced Chatbot (Sentiment): I sense a positive tone!")
    elif sentiment < -0.3:
        print("Advanced Chatbot (Sentiment): I detect a somewhat negative sentiment.")

    # Get response from the advanced NLP logic
    # In a real system, `chat_history_vectors` would be actual embeddings
    response = advanced_nlp_response(user_input, list(conversation_history))
    print(f"Advanced Chatbot: {response}")
    add_to_history("Chatbot", response)
