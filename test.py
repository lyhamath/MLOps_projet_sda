from app import model_pred

new_data = {'credit_lines_outstanding':4522930,
            'loan_amt_outstanding': 5,
            'total_debt_outstanding':2761.049506,
            'income':16620.80342,
            'years_employed':4,
            'fico_score':627,
            }


def test_predict():
    prediction = model_pred(new_data)
    assert prediction == 1, "incorrect prediction"
