from flask import Flask, render_template, request as req
from client import Client

app = Flask(__name__, static_url_path='')
my_client = Client()

@app.route('/')
def index():
    return render_template('index.html.j2')

@app.route('/peninsula/eval', methods=['POST', 'GET'])
def predict():
    # return my_client.predict(req.form['tweet'])
    pred = my_client.predict(req.form['tweet'])
    predictions = pred[0]
    return render_template('prediction.html.j2', \
        tweet=req.form['tweet'], pri=predictions[0], morena=predictions[1], pan=predictions[2])

app.run(host='127.0.0.1', port=8080, debug=True)
