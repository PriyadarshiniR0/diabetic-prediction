from flask import Flask, render_template, request
import pickle, modtrain
import numpy as np

modtrain.trainmodel()

# Load the Random Forest CLassifier model
filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
     if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)
        if my_prediction == 1:
            pred = "You have Diabetes, please consult a Doctor."
            return render_template('result.html',pregnancy=preg, gluc=glucose, blood=bp, skin=st, ins=insulin, bmi=bmi, dpf=dpf, age=age, prediction_text=pred)
        elif my_prediction == 0:
            pred = "You don't have Diabetes."
            return render_template('result.html',pregnancy=preg, gluc=glucose, blood=bp, skin=st, ins=insulin, bmi=bmi, dpf=dpf, age=age, prediction_text=pred)
        # return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=7770)