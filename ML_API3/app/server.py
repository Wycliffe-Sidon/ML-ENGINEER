import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from fastapi import FastAPI

app = FastAPI()
# Load the trained model and vectorizer
try:
    lr = joblib.load('app/lr.joblib')
    vectorizer = joblib.load('app/vectorizer.joblib')
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("Error: Model or vectorizer files not found. Make sure 'spam_detection_model.pkl' and 'vectorizer.pkl' are in the same directory.")

def predict_spam(email_message):
    """
    Predicts whether an email message is spam or not.

    Args:
        email_message (str): The email message to predict.

    Returns:
        str: 'spam' or 'ham'.
    """
    if lr is None or vectorizer is None:
        return "Error: Model or vectorizer not loaded."

    # Preprocess the email message
    email_vector = vectorizer.transform([email_message])

    # Make the prediction
    prediction = lr.predict(email_vector)

    # Decode the prediction (0 for ham, 1 for spam)
    if prediction[0] == 1:
        return 'spam'
    else:
        return 'ham'

if __name__ == '__main__':
    # Example usage
    test_email = "Congratulations! You've won a free cruise!"
    prediction = predict_spam(test_email)
    print(f"The email '{test_email}' is predicted as: {prediction}")

    test_email_2 = "Meeting reminder for tomorrow at 10 AM."
    prediction_2 = predict_spam(test_email_2)
    print(f"The email '{test_email_2}' is predicted as: {prediction_2}")