from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Charger le modèle avec joblib
model = joblib.load('Model_Logistic.pkl')

def model_pred(features):
    test_data = pd.DataFrame([features])
    prediction = model.predict(test_data)
    return int(prediction[0])


@app.route("/", methods=["GET"])
def Home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        credit_lines_outstanding = int(request.form["credit_lines_outstanding"])
        loan_amt_outstanding = float(request.form["loan_amt_outstanding"])
        total_debt_outstanding = float(request.form["total_debt_outstanding"])
        income = float(request.form["income"])
        years_employed = int(request.form["years_employed"])
        fico_score= int(request.form["fico_score"])
        prediction = model.predict(
            [[credit_lines_outstanding, loan_amt_outstanding, total_debt_outstanding, income, years_employed, fico_score]]
        )

        if prediction[0] == 1:
            return render_template(
                "index.html",
                prediction_text="le client est insolvable",
            )

        else:
            return render_template(
                "index.html", prediction_text="Le client est solvable"
            )

    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
