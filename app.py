from flask import Flask, request, render_template
import numpy as np
import pickle

# importing model
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Get input data from form
        N = request.form['Nitrogen']
        P = request.form['Phosporus']
        K = request.form['Potassium']
        temp = request.form['Temperature']
        humidity = request.form['Humidity']
        ph = request.form['Ph']
        rainfall = request.form['Rainfall']

        # Check if any field is empty
        if not all([N, P, K, temp, humidity, ph, rainfall]):
            raise ValueError("Please provide valid inputs")

        # Convert input to float and create feature list
        features = [float(N), float(P), float(K), float(temp), float(humidity), float(ph), float(rainfall)]

        # Edge Case: If all inputs are zero
        if all([f == 0 for f in features]):
            result = "No suitable crop can be cultivated with these inputs."
            return render_template('index.html', result=result)

        # Outlier Check: Values outside realistic ranges
        if not (0 <= float(N) <= 200 and 0 <= float(P) <= 200 and 0 <= float(K) <= 500 and -10 <= float(temp) <= 100 and
                0 <= float(humidity) <= 100 and 0 <= float(ph) <= 14 and 0 <= float(rainfall) <= 1000):
            raise ValueError("Inputs out of range")

        # Scale the features using only MinMaxScaler
        single_pred = np.array(features).reshape(1, -1)
        transformed_features = ms.transform(single_pred)

        # Predict using the trained model
        prediction = model.predict(transformed_features)

        # Crop prediction dictionary
        crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                     8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                     14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                     19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

        # Map prediction to crop name
        crop = crop_dict.get(prediction[0], "Unknown crop")
        result = "{} is the best crop to be cultivated right there".format(crop)

    except ValueError as e:
        result = str(e)
    except Exception as e:
        result = "An error occurred: " + str(e)

    return render_template('index.html', result=result)


# python main
if __name__ == "__main__":
    app.run(debug=True)
