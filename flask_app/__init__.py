from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    model = pickle.load(open('modeling.pkl', 'rb'))
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8]])
    pred = model.predict(arr)
    return render_template('result.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)