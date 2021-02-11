from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)
classifier = pickle.load(open('trained_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/decide', methods=['GET', 'POST'])
def homepage():
    if request.method == 'POST':
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        pred = classifier.predict(final_features)

        if pred[0] == 0:
            return render_template('input.html', output="The patient does not have a heart disease !!")
        else:
            return render_template('input.html', output="The patient has a heart disease !!")
    else:
        return render_template('input.html')

if __name__ == '__main__':
    app.run(debug=True)