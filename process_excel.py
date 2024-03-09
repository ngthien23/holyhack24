import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

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

    df['pred.amount'] = (df['BETAALD BEDRAG - extracted from invoice'] - df['BETAALD BEDRAG - declared ']).apply(lambda x: 1 if x < 0 else 0)

    df['pred.supplier'] = (df['LEVERANCIER - EXTRACTED'] != df['LEVERANCIER - DECLARED']).astype(int)

    df['pred.date'] = (df['DATUM FACTUUR - extracted'] != df['DATUM FACTUUR - DECLARED']).astype(int)

    # Filter rows where "category" column has value 0
    df2 = df[df['category'] == 0]

    # Save the filtered DataFrame to a CSV file
    df2.to_csv('cat_data.csv', index=False)

    # Train model on positive targets
    X = df2['BESCHRIJVING DECLARATIE']
    y = df2['KOSTENRUBRIEK - declared']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train a Multinomial Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X_train_tfidf, y_train)

    # Make predictions on the test set
    predictions = classifier.predict(X_test_tfidf)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    classification_rep = classification_report(y_test, predictions)

    print(f'Accuracy: {accuracy:.2f}')
    print('\nClassification Report:')
    print(classification_rep)

    # Use model to predict on negative targets
    df_category_1 = df[df['category'] == 1]

    # Extract text data and labels
    X_category_1 = df_category_1['BESCHRIJVING DECLARATIE']
    y_category_1 = df_category_1['KOSTENRUBRIEK - declared']

    # Vectorize the text data using the same TF-IDF vectorizer
    X_category_1_tfidf = vectorizer.transform(X_category_1)

    # Make predictions on the instances where 'category' has value 1
    predictions_category_1 = classifier.predict(X_category_1_tfidf)

    # Evaluate the model on these instances
    accuracy_category_1 = accuracy_score(y_category_1, predictions_category_1)
    classification_rep_category_1 = classification_report(y_category_1, predictions_category_1)

