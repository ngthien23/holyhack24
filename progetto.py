import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn import tree


df = pd.read_excel("costlines-20180101-20221231-v2.xlsx")
print(df["BESCHRIJVING DECLARATIE"])

def vectorize_text(text_data):
    # Create a CountVectorizer
    vectorizer = CountVectorizer()

    # Fit and transform the text data
    vectorized_data = vectorizer.fit_transform(text_data)

    return vectorized_data, vectorizer

vectorized_data, vectorizer = vectorize_text(df["BESCHRIJVING DECLARATIE"])

#print(vectorized_data)


X_train, X_test, y_train, y_test = train_test_split(vectorized_data, df["eligible"], random_state=0)
#print(X_train,y_train)
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

print(len(y_pred),sum(y_pred),y_pred)