import joblib

# Load trained model
model = joblib.load('trained_wine_classifier_model.pkl')

# Provide features for wine quality you want to value
wine_to_value =[
    1.0,    # fixed acidity
    0.5,    # volatile acidity
    0.5,    # citric acid
    15.0,   # residual sugar
    0.5,    # chlorides
    50.0,   # free sulfur dioxide
    150.0,  # total sulfur dioxide
    2.0,    # density
    3.5,    # pH
    0.5,    # sulphates
    10.0,   # alcohol
]

wines_to_value = [
    wine_to_value
]

# Run the model to predict the value for each wine in the wines_to_value array
predicted_wine_qualities = model.predict(wines_to_value)

# Since we are only predicting the price of one type of wine, let's look at the first prediction returned
predicted_value = predicted_wine_qualities[0]

print("This wine has an estimated quality of {:,.2f}".format(predicted_value))
