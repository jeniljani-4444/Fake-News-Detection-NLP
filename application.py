import streamlit as st 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re




st.set_page_config(
    page_title="Fake News Detection",
    page_icon=":newspaper:",
    initial_sidebar_state="expanded",
)

hide_streamlit_footer = """

<style>
    footer {visibility:hidden;}
</style>

"""

st.markdown(hide_streamlit_footer,unsafe_allow_html=True)


port_stem = PorterStemmer()
vectorization = TfidfVectorizer()

vector_form = pickle.load(open('vector.pkl','rb'))
load_model = pickle.load(open('model.pkl','rb'))

def text_preprocessing(content):
    clean_text = re.sub("[^a-zA-Z]"," ",content)
    clean_text  = clean_text.lower()
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

if __name__ == '__main__':
    st.title("Fake News Classification :newspaper:")
    st.subheader("Input the news content below")
    sentence = st.text_area("Enter your news",height=200)
    predict_btn = st.button("Predict")

    if predict_btn:
        prediction_class = fake_news(sentence)
        print(prediction_class)

        if prediction_class==[0]:
            st.error("The news is fake")
        else:
            st.success("The news is real")


footer = '''
<div style="position: fixed; bottom: 0; left: 0; width: 100%; text-align: center; padding: 10px; background-color: #000; color: #fff;">
    <p>Made with &#x2764;&#xFE0F; by Jenil Jani</p>
</div>
'''

st.markdown(footer, unsafe_allow_html=True)