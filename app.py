

from flask import Flask, render_template, request

import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load your pre-trained model and TF-IDF vectorizer
rating_model = joblib.load('rating_model.pkl')
tfidf_vectorizer = joblib.load('vectorizer.pkl')

# Set up NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    review_text = ""

    if request.method == 'POST':
        review_text = request.form['review_text']
        
        # Preprocess the new review text
        new_review_tokens = nltk.word_tokenize(review_text.lower())
        new_review_tokens = [word for word in new_review_tokens if word.isalnum()]
        new_review_tokens = [word for word in new_review_tokens if word not in stop_words]
        new_review_text = ' '.join(new_review_tokens)
        
        # Vectorize the new review text
        new_review_vector = tfidf_vectorizer.transform([new_review_text])
        
        # Make rating prediction
        rating_prediction = rating_model.predict(new_review_vector)
        
        # Clamp the predicted rating to the range of 1 to 5
        prediction = max(1, min(5, rating_prediction[0]))
    
    return render_template('index.html', prediction=prediction, review_text=review_text)

if __name__ == '__main__':
    app.run(debug=True)
