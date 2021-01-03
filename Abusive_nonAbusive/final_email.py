# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 13:08:38 2021

@author: yasir Arafath
"""
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer
from joblib import dump
import joblib

email=pd.read_csv("C:\\Users\\yasir Arafath\\Downloads\\emails.csv")
email.head()
email.drop(email.columns[[0,1,2]], axis=1, inplace=True)

email['content'].head()

#preprocessing

# \r and \n
email['Content_Parsed_1'] = email['content'].str.replace("\r", " ")
email['Content_Parsed_1'] = email['Content_Parsed_1'].str.replace("\n", " ")
email['Content_Parsed_1'] = email['Content_Parsed_1'].str.replace("    ", " ")
# " when quoting text
email['Content_Parsed_1'] = email['Content_Parsed_1'].str.replace('"', '')
# Lowercasing the text
email['Content_Parsed_2'] = email['Content_Parsed_1'].str.lower()


punctuation_signs = list("?:!.,;")
email['Content_Parsed_3'] = email['Content_Parsed_2']

for punct_sign in punctuation_signs:
    email['Content_Parsed_3'] = email['Content_Parsed_3'].str.replace(punct_sign, '')
email['Content_Parsed_4'] = email['Content_Parsed_3'].str.replace("'s", "")

# Downloading punkt and wordnet from NLTK
nltk.download('punkt')
print("------------------------------------------------------------")
nltk.download('wordnet')


# Saving the lemmatizer into an object
wordnet_lemmatizer = WordNetLemmatizer()
nrows = len(email)
lemmatized_text_list = []

for row in range(0, nrows):
    
    # Create an empty list containing lemmatized words
    lemmatized_list = []
    
    # Save the text and its words into an object
    text = email.loc[row]['Content_Parsed_4']
    text_words = text.split(" ")

    # Iterate through every word to lemmatize
    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
        
    # Join the list
    lemmatized_text = " ".join(lemmatized_list)
    
    # Append to the list containing the texts
    lemmatized_text_list.append(lemmatized_text)


email['Content_Parsed_5'] = lemmatized_text_list
email['Content_Parsed_5'].head()
# Downloading the stop words list
nltk.download('stopwords')
# Loading the stop words in english
stop_words = list(stopwords.words('english'))

email['Content_Parsed_6'] = email['Content_Parsed_5']

for stop_word in stop_words:

    regex_stopword = r"\b" + stop_word + r"\b"
    email['Content_Parsed_6'] = email['Content_Parsed_6'].str.replace(regex_stopword, '')
email['Content_Parsed_6'].head()
email['Content_Parsed_6']=email['Content_Parsed_6'].apply(lambda x:re.sub(r'[^a-zA-z]+', " ",x))
email['Content_Parsed_6'].head()
email['label'] = email['Class'].map({'Abusive': 0, 'Non Abusive': 1})

email.head()
email.columns
list_columns = ["content", "Class", "Content_Parsed_6", "label"]
email = email[list_columns]
email = email.rename(columns={'Content_Parsed_6': 'Content_Parsed'})

# Count Vectorizer
count_vect = CountVectorizer(max_features=5000)
X = count_vect.fit_transform(email["Content_Parsed"]).toarray()


# Train and Test Split
X_train, X_test, y_train, y_test = train_test_split(X, email["label"], test_size = 0.1, random_state = 42)

# Oversampling
oversample = SMOTE(random_state = 42, sampling_strategy = 'minority')
  


X_train_oversample, y_train_oversample = oversample.fit_sample(X_train, y_train)



clf = MultinomialNB()
clf.fit(X_train_oversample, y_train_oversample)
clf.score(X_test,y_test)


# MODEL SAVED TO ANOTHER FILE
joblib.dump(clf, 'final_model.pkl')
joblib.dump(count_vect, 'vector.pkl')
