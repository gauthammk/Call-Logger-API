from flask import Flask
from .test import testFunction
# from prediction import preprocessing
app = Flask(__name__)


@app.route('/')
def index():
    return 'Hello, world!'


# @app.route('/prediction')
# def prediction():
#     result = preprocessing('sample_calls.csv', 'sample_SMS.csv')
#     return result

@app.route('/test')
def test():
    return testFunction()
