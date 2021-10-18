from flask import Flask, jsonify, request
from flask_restful import Api, Resource

import re
import nltk
import pandas as pd
from flask import Flask, render_template, request
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from openpyxl import load_workbook
import numpy as np
wordnet = WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
api = Api(app)

#class Solution(Resource):
@app.route('/solution', methods=['POST'])
def processing():
    postedData = request.get_json()

    question = postedData["question"]

    df = pd.read_csv('data3.csv')
    df = df.drop('No.', axis=1)
    df = df.rename(columns={'Questions/ Issue ': 'Questions', 'Answer /Steps to Resolve': 'Answers'})

    cleaned_question = []
    for i in range(len(df['Questions'])):
        review = re.sub('[^a-zA-Z0-9]', ' ', df['Questions'][i])
        review = review.lower()
        review = review.split()
        # review = [wordnet.lemmatize(word) for word in review]#add stop word code
        review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        # print(review)
        cleaned_question.append(review)

    df['cleaned_questions'] = cleaned_question

    cleaned_data_list = list(df['cleaned_questions'])

    search_terms = question

    review = re.sub('[^a-zA-Z0-9]', ' ', search_terms)
    review = review.lower()
    review = review.split()
    # review = [wordnet.lemmatize(word) for word in review] #add stop word
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    search_terms = ' '.join(review)

    doc_vectors = TfidfVectorizer()
    doc_vectors = doc_vectors.fit_transform([search_terms] + cleaned_data_list)
    cosine_similarities = linear_kernel(doc_vectors[0:1], doc_vectors[1:]).flatten()
    df['cosine_score'] = cosine_similarities

    highest_score = 0
    highest_score_index = 0
    for i, score in enumerate(cosine_similarities):
        if highest_score < score:
            highest_score = score
            highest_score_index = i

    # if highest_score < 0.1:
    #   highest_score_index = 127

    most_similar_question = df['Questions'][highest_score_index]
    most_similar_answer = df['Answers'][highest_score_index]
    # print("Your Query: ", most_similar_question, "\n\n", "Here is your solution: ", most_similar_answer)
    if highest_score < 0.1:
        prediction_answer = "Please enter a valid question"
        prediction_question = "No match found"
    else:
        prediction_answer = most_similar_answer
        prediction_question = most_similar_question

    #retJson = {
    #    most_similar_answer
    #}
    return (most_similar_answer)

#api.add_resource(Solution, "/solution")

@app.route('/')
def hello_world():
    return "Hello World!"


if __name__=="__main__":
    app.run()






