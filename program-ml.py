from flask import Flask, request, render_template
from utils import tokenize
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

tfidf_model = pickle.load(open('tfidf_vectorizer_train.pkl', 'rb'))
logistic_toxic_model = pickle.load(open('logistic_toxic.pkl', 'rb'))
logistic_severe_toxic_model = pickle.load(open('logistic_severe_toxic.pkl', 'rb'))
logistic_identity_hate_model = pickle.load(open('logistic_identity_hate.pkl', 'rb'))
logistic_insult_model = pickle.load(open('logistic_insult.pkl', 'rb'))
logistic_obscene_model = pickle.load(open('logistic_obscene.pkl', 'rb'))
logistic_threat_model = pickle.load(open('logistic_threat.pkl', 'rb'))


@app.route('/')
def my_form():
    return render_template('main.html')


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']

    comment_term_doc = tfidf_model.transform([text])

    dict_preds = {}

    dict_preds['pred_toxic'] = logistic_toxic_model.predict_proba(comment_term_doc)[:, 1][0]
    dict_preds['pred_severe_toxic'] = logistic_severe_toxic_model.predict_proba(comment_term_doc)[:, 1][0]
    dict_preds['pred_identity_hate'] = logistic_identity_hate_model.predict_proba(comment_term_doc)[:, 1][0]
    dict_preds['pred_insult'] = logistic_insult_model.predict_proba(comment_term_doc)[:, 1][0]
    dict_preds['pred_obscene'] = logistic_obscene_model.predict_proba(comment_term_doc)[:, 1][0]
    dict_preds['pred_threat'] = logistic_threat_model.predict_proba(comment_term_doc)[:, 1][0]

    for k in dict_preds:
        perc = dict_preds[k] * 100
        dict_preds[k] = "{0:.2f}%".format(perc)

    return render_template('main.html', text=text,
                           pred_toxic=dict_preds['pred_toxic'],
                           pred_severe_toxic=dict_preds['pred_severe_toxic'],
                           pred_identity_hate=dict_preds['pred_identity_hate'],
                           pred_insult=dict_preds['pred_insult'],
                           pred_obscene=dict_preds['pred_obscene'],
                           pred_threat=dict_preds['pred_threat'])


if __name__ == '__main__':

    app.run(debug=True)
