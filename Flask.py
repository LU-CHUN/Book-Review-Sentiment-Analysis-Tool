from flask import Flask, render_template, request, jsonify
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import joblib
import nltk

app = Flask(__name__)

svm_model = joblib.load('svm_sentiment_model.pkl')
sid = SentimentIntensityAnalyzer()



def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text



def perform_sentiment_analysis(text):

    cleaned_text = clean_text(text)


    sentiment_score = sid.polarity_scores(cleaned_text)['compound']


    hashtag_sentiments = [{'hashtag': 'example', 'sentiment': sentiment_score}]

    return sentiment_score, hashtag_sentiments


@app.route('/')
def home():
    return render_template('web.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.json['text']
    sentiment_score, hashtag_sentiments = perform_sentiment_analysis(text)
    return jsonify({
        'overall_sentiment_score': sentiment_score,
        'hashtag_sentiments': hashtag_sentiments
    })


if __name__ == '__main__':
    app.run(debug=True)