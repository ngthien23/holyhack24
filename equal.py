from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import pandas as pd
import re

se=SnowballStemmer("english")
df = pd.read_excel("output_file.xlsx")

x=df["LEVERANCIER - EXTRACTED"]
y=df["LEVERANCIER - DECLARED"]

def clean_message(message_list):
    ms=[]
    for word in message_list:
        ms.append(se.stem(word))
    return ms 

def lever_equals(x,y):
    if x==y:
        return 0
    x=re.sub(r'[^\w\s]','',x)
    y=re.sub(r'[^\w\s]','',y)
    split_x=clean_message(word_tokenize(x.lower()))
    split_y=clean_message(word_tokenize(y.lower()))
    
    if len(split_x)==len(split_y):
        for i in range(len(split_x)):
            if split_x[i][0]!=split_y[i][0]:
                return 1
        return 0
    else:
        mini=min(len(split_x),len(split_y))
        for i in range(mini):
            if split_x[i]!=split_y[i]:
                return 1
        return 0
    return 1

#to check if it's working
l=[]

for i in range(len(x)):
    l.append(lever_equals(x[i],y[i]))

print(l,sum(l))