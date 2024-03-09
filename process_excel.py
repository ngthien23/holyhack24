import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk import word_tokenize
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

def process_excel(uploaded_file):

    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(uploaded_file)
    df['KOSTENRUBRIEK - declared'] = df['KOSTENRUBRIEK - declared'].str.lower()
    df['FLC: REDEN VERWERPING'] = df['FLC: REDEN VERWERPING'].str.lower()
    df['BESCHRIJVING DECLARATIE'] = df['BESCHRIJVING DECLARATIE'].str.lower()

    # Extract unique flags from the "FLC: REDEN VERWERPING" column
    all_flags = df["FLC: REDEN VERWERPING"].str.split('|', expand=True).stack().str.strip().unique()

    # Print the unique flags
    print("Unique Flags:")
    for flag in all_flags:
        print(flag)
    
    # Define the phrases to check for in the specified column
    phrases = {
        "FLC: REDEN VERWERPING": {
            "double declaration": "double",
            "too late": "late",
            "correct invoiced amount": "amount",
            "supplier": "supplier",
            "wrong category": "category",
            "eligible": "eligible"
        }
    }

    # Iterate through the phrases and update the corresponding columns
    for column_name, conditions in phrases.items():
        for phrase, new_column in conditions.items():
            df[new_column] = df[column_name].str.contains(phrase, case=False, na=False).astype(int)
    
    # Convert the "DATUM FACTUUR - DECLARED" column to datetime format
    df['DATUM FACTUUR - DECLARED'] = pd.to_datetime(df['DATUM FACTUUR - DECLARED'], errors='coerce')

    # Create the "latest" column by moving the month 6 months forward
    df['latest'] = df['DATUM FACTUUR - DECLARED'] + pd.DateOffset(months=6)

    # Convert date columns to datetime objects
    df['latest'] = pd.to_datetime(df['latest'])
    df['DECLARATIEDATUM (can be assumed to be close to payment date)'] = pd.to_datetime(df['DECLARATIEDATUM (can be assumed to be close to payment date)'])

    # Calculate the time difference and create a new column 'timediff'
    df['pred.late'] = (df['DECLARATIEDATUM (can be assumed to be close to payment date)'] - df['latest']).dt.days > 0
    df['pred.late'] = df['pred.late'].astype(int)

    # Equality check
    df['pred.amount'] = (df['BETAALD BEDRAG - extracted from invoice'] - df['BETAALD BEDRAG - declared ']).apply(lambda x: 1 if x < 0 else 0)
    df['pred.supplier'] = (df['LEVERANCIER - EXTRACTED'] != df['LEVERANCIER - DECLARED']).astype(int)
    df['pred.date'] = (df['DATUM FACTUUR - extracted'] != df['DATUM FACTUUR - DECLARED']).astype(int)

    # Model 1
    #classifier = joblib.load('text_classifier_model.joblib')
    #X_new = df['BESCHRIJVING DECLARATIE']
    #vectorizer = TfidfVectorizer(stop_words='english')
    #X_new_tfidf = vectorizer.transform(X_new)
    #predictions_new = classifier.predict(X_new_tfidf)
    #df['category.pred'] = predictions_new

    return df

