from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, template_folder='Templates')
model = pickle.load(open('model.pkl', 'rb'))


@app.route("/", methods=["GET"])
def Home_page():
    return render_template("index.html")


standard_to = StandardScaler()


@app.route("/predict", methods=["POST"])
def prediction_page():
    if request.method == "POST":
        height = int(request.form["height"])

        prediction = model.predict([[height]])
        return render_template('result.html', result=prediction)

    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
