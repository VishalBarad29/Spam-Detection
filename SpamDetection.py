import pandas as pd 
from sklearn.model_selection import train_test_split #to split the data into train and test
from sklearn.feature_extraction.text import CountVectorizer  #to convert the text data into numerical data
from sklearn.naive_bayes import MultinomialNB

import streamlit as st
data = pd.read_csv("spam.csv")  #Read the data from CSV file

# print(data.head())
# print(data.shape)

data.drop_duplicates(inplace=True) #Remove the duplicates from the dataset

data['Category'] = data['Category'].replace(['ham','spam'],['Not Spam','Spam']) #replace ham with not spam in the category column

msg = data['Message']

cat = data['Category']

(msg_train, msg_test, cat_train, cat_test) = train_test_split(msg, cat, test_size=0.2) #split the data into training and testing (here 20% of data will be used for testing)

cv = CountVectorizer(stop_words='english') #we will eliminate the not important words from the data like (a,I,am,he)., common words in English
features = cv.fit_transform(msg_train) #convert the train data into numerical format

#Creating model

model = MultinomialNB()

model.fit(features, cat_train) #Train our model

#Test our model
features_test = cv.transform(msg_test)
print(model.score(features_test, cat_test))  #for finding accuracy of our model

#predict the data in real-Time
def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result

st.header('Spam Detection')

input_mess = st.text_input('Enter Message here')

if st.button('Validate'):
    output = predict(input_mess)
    # st.markdown(output)
    result = output[0]
    if result == "Spam":
        st.error("🚨 Spam Detected!")
    else:
        st.success("✅ Not Spam")



