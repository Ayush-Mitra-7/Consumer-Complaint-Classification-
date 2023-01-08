from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

cv = pickle.load(open('cv.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['post'])
def take_input():
    text_input = request.form.get('complaint')
    data = cv.transform([text_input]).toarray()
    output = model.predict(data)
    return render_template('index.html', data=output[0])


if __name__ == '__main__':
    app.run(debug=True)
