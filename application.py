from flask import Flask, render_template, request
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

application = Flask(__name__)

port_stem = PorterStemmer()
vectorization = TfidfVectorizer()

vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

def text_preprocessing(content):
    clean_text = re.sub("[^a-zA-Z]", " ", content)
    clean_text = clean_text.lower()
    clean_text = clean_text.split()
    clean_text = [port_stem.stem(word) for word in clean_text if word not in stopwords.words('english')]
    clean_text = ' '.join(clean_text)
    return clean_text

def fake_news(news):
    news = text_preprocessing(news)
    input_data = [news]
    vector_form1 = vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        sentence = request.form['news_text']
        prediction_class = fake_news(sentence)

        if prediction_class == [0]:
            result = "The news is fake"
        else:
            result = "The news is real"

        return render_template('index.html', result=result, news_text=sentence)

if __name__ == '__main__':
    application.run(host="0.0.0.0",port=8000,debug=True)
