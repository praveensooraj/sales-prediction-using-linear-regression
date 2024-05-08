from flask import Flask, request, jsonify
import joblib

# Load the trained model
model = joblib.load('your_trained_model.pkl')

# Initialize Flask application
app = Flask(__name__)

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.get_json()

    # Make predictions
    predictions = model.predict(data['features'])

    # Return predictions
    return jsonify({'predictions': predictions.tolist()})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
