import array
from flask import Flask,render_template,request
import numpy as np
import pickle

classifier = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/',methods=['GET'])
def breast_cancer():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def predict():
    data1 = request.form['mean_radius']
    data2 = request.form['texture_mean']
    data3 = request.form['smoothness_mean']
    data4 = request.form['compactness_mean']
    data5 = request.form['symmetry_mean']
    data6 = request.form['texture_se']
    data7 = request.form['compactness_se']
    data8 = request.form['symmetry_se']
    data9 = request.form['symmetry_worst']
    list_data = np.array([[float(data1), float(data2), float(data3), float(data4),float(data5),float(data6),float(data7),float(data8),float(data9)]])
      
    pred = classifier.predict(list_data)
    return render_template('predict.html', data=pred)


if __name__ == '__main__':
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=8080)
    app.run(debug=True)