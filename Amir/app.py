from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

label_encoders = joblib.load("label_encoders.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            input_data = {
                'Company Name': request.form['company'],
                'Model Name': request.form['model'],
                'Model Year': int(request.form['year']),
                'Location': request.form['location'],
                'Mileage': int(request.form['mileage']),
                'Engine Type': request.form['engine_type'],
                'Engine Capacity': int(request.form['engine_capacity']),
                'Color': request.form['color'],
                'Assembly': request.form['assembly'],
                'Body Type': request.form['body_type'],
                'Transmission Type': request.form['transmission'],
                'Registration Status': request.form['registration']
            }

            df = pd.DataFrame([input_data])

            for col, le in label_encoders.items():
                if col in df:
                    df[col] = le.transform(df[col])

            scaled = scaler.transform(df)
            prediction = model.predict(scaled)
            return render_template("index.html", prediction=int(prediction[0]))

        except Exception as e:
            return f"Error: {e}"

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)