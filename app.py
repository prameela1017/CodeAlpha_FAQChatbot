from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from faq import faq_data
import re

app = Flask(__name__)

# 🔹 Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# 🔹 Prepare data
questions = [clean_text(q) for q in faq_data.keys()]
answers = list(faq_data.values())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# 🔹 Chat history
chat_history = []

@app.route('/', methods=['GET', 'POST'])
def index():
    global chat_history

    if request.method == 'POST':
        user_input = request.form['question']

        cleaned_input = clean_text(user_input)
        user_vec = vectorizer.transform([cleaned_input])

        similarity = cosine_similarity(user_vec, X)
        index = similarity.argmax()

        # 🔥 Unknown question handling
        if similarity[0][index] < 0.3:
            response = "Sorry, I don't understand your question."
        else:
            response = answers[index]

        # 🔹 Save chat history
        chat_history.append({"sender": "You", "text": user_input, "type": "user"})
        chat_history.append({"sender": "Bot", "text": response, "type": "bot"})

    return render_template('index.html', chat=chat_history)

if __name__ == '__main__':
    app.run(debug=True, port=5001)