import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer


vectorizer = TfidfVectorizer(stop_words='english')

def process_excel(uploaded_file):

    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(uploaded_file)
    df['KOSTENRUBRIEK - declared'] = df['KOSTENRUBRIEK - declared'].str.lower()
    df['FLC: REDEN VERWERPING'] = df['FLC: REDEN VERWERPING'].str.lower()
    df['BESCHRIJVING DECLARATIE'] = df['BESCHRIJVING DECLARATIE'].str.lower()

    # Extract unique flags from the "FLC: REDEN VERWERPING" column
    all_flags = df["FLC: REDEN VERWERPING"].str.split('|', expand=True).stack().str.strip().unique()
    
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
    
    # Double declaration check
    df['pred.double'] = 0

    # Check if an entry has the same values in both columns 'REFERENTIE FACTUUR' and 'DATUM FACTUUR - DECLARED'
    duplicate_entries = df[df.duplicated(['REFERENTIE FACTUUR', 'DATUM FACTUUR - DECLARED'], keep=False)]

    for idx, group in duplicate_entries.groupby(['REFERENTIE FACTUUR', 'DATUM FACTUUR - DECLARED']):
        # Find the earliest value in 'DECLARATIEDATUM (can be assumed to be close to payment date)' for this entry
        earliest_date = group['DECLARATIEDATUM (can be assumed to be close to payment date)'].min()
        
        # Set 'pred.double' to 1 for entries other than the one with the earliest date
        df.loc[group.index, 'pred.double'] = (group['DECLARATIEDATUM (can be assumed to be close to payment date)'] != earliest_date).astype(int)

    # Convert the "DATUM FACTUUR - DECLARED" column to datetime format
    df['DATUM FACTUUR - DECLARED'] = pd.to_datetime(df['DATUM FACTUUR - DECLARED'], errors='coerce')

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

    # Amount check
    df['pred.amount'] = (df['BETAALD BEDRAG - extracted from invoice'] - df['BETAALD BEDRAG - declared ']).apply(lambda x: 1 if x < 0 else 0)

    # Supplier check
    se = SnowballStemmer("english")

    x = df["LEVERANCIER - EXTRACTED"]
    y = df["LEVERANCIER - DECLARED"]

    def clean_message(message_list):
        ms = []
        for word in message_list:
            ms.append(se.stem(word))
        return ms 

    def lever_equals(x, y):
        if x == y:
            return 0
        x = re.sub(r'[^\w\s]', ' ', x)
        y = re.sub(r'[^\w\s]', ' ', y)
        split_x = clean_message(word_tokenize(x.lower()))
        split_y = clean_message(word_tokenize(y.lower()))
        
        if len(split_x) == len(split_y):
            for i in range(len(split_x)):
                if split_x[i][0] != split_y[i][0]:
                    return 1
            return 0
        elif len(split_x)!=0 and len(split_y)!=0:
            mini = min(len(split_x), len(split_y))
            for i in range(mini):
                if split_x[i] != split_y[i]:
                    return 1
            return 0
        return 1

    # Create a new column 'pred.supplier' in df and store the results
    df['pred.supplier'] = [lever_equals(x[i], y[i]) for i in range(len(x))]

    # Category check
    # Filter rows where "category" column has value 0
    df2 = df[df['category'] == 0].copy()  # Make a copy of the filtered DataFrame

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

    # Make predictions on the training set
    train_predictions = classifier.predict(X_train_tfidf)

    # Update 'pred.category' in the original DataFrame for the training set
    df2.loc[df2['BESCHRIJVING DECLARATIE'].index, 'pred.category'] = (train_predictions != y_train).astype(int)

    # Make predictions on the test set
    test_predictions = classifier.predict(X_test_tfidf)

    # Update 'pred.category' in the original DataFrame for the test set
    df2.loc[X_test.index, 'pred.category'] = (test_predictions != y_test).astype(int)

    # Use model to predict on positive targets
    df_category_1 = df[df['category'] == 1].copy()  # Make a copy to avoid modifying the original DataFrame

    # Extract text data and labels
    X_category_1 = df_category_1['BESCHRIJVING DECLARATIE']
    y_category_1 = df_category_1['KOSTENRUBRIEK - declared']

    # Vectorize the text data using the same TF-IDF vectorizer
    X_category_1_tfidf = vectorizer.transform(X_category_1)

    # Make predictions on the instances where 'category' has value 1
    predictions_category_1 = classifier.predict(X_category_1_tfidf)

    # Update 'pred.category' in the original DataFrame
    df_category_1['pred.category'] = 1-(predictions_category_1 == y_category_1).astype(int)

    # Merge
    df = pd.concat([df2, df_category_1], ignore_index=True)
    
    # Eligible check
    df_eli = pd.read_excel(eli_data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df_eli['BESCHRIJVING DECLARATIE'], df_eli['eligible'], test_size=0.2, random_state=42)

    # Create a pipeline with TF-IDF vectorizer, stopwords removal, and Decision Tree classifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),  # Using TF-IDF vectorizer with stopwords removal
        ('classifier', DecisionTreeClassifier(random_state=42, class_weight="balanced"))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)

    # Evaluate the model on the test set
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on the test set: {accuracy:.2f}")

    # Assuming 'df' is your DataFrame for testing
    # Make predictions on the 'df' DataFrame
    df['pred.eligible'] = pipeline.predict(df['BESCHRIJVING DECLARATIE'])

    return df

