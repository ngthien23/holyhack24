import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk import word_tokenize

s=set(stopwords.words("english"))
df = pd.read_excel("dataset-v2.xlsx")
print(df["BESCHRIJVING DECLARATIE"])


def vectorize_text(text_data):
    # Create a CountVectorizer
    vectorizer = CountVectorizer()

    #taking out the stopwords from the descriptions
    text_data = [preprocess_text(text) for text in text_data]

    # Fit and transform the description
    vectorized_data = vectorizer.fit_transform(text_data)

    return vectorized_data, vectorizer

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text
    words = word_tokenize(text)
    
    # Remove English stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    
    # Join the filtered words back into a string
    processed_text = ' '.join(filtered_words)
    
    return processed_text


vectorized_data, vectorizer = vectorize_text(df["BESCHRIJVING DECLARATIE"])
#print(vectorized_data)

X_train, X_test, y_train, y_test = train_test_split(vectorized_data, df["eligible"], random_state=0)
#print(X_train,y_train)
clf = DecisionTreeClassifier(random_state=0,class_weight="balanced")

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()

print(y_pred,X_test)