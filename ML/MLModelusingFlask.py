import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

#read the dataset
data = pd.read_csv('dataset/twitter_sentiments.csv')

#view the top row of datasedata.head()
data.head()

#train test split
train, test = train_test_split(data, test_size = .2, stratify = data['label'], random_state=21)

#get the shape of the train and test split
train.shape, test.shape

# create a TF-IDF vectorizer object
tfidf_vectorizer = TfidfVectorizer(lowercase=True, max_features=1000, stop_words=ENGLISH_STOP_WORDS)
# fit the object with the trainin data tweets
tfidf_vectorizer.fit(train.tweet)

# transform the train and test data
train_idf = tfidf_vectorizer.transform(train.tweet)
test_idf = tfidf_vectorizer.transform(test.tweet)

# create the object of LinearRegression model
model_LR = LogisticRegression()

#fit the model with the training data
model_LR.fit(train_idf, train.label)

#fit the model with the training data
predict_train = model_LR.predict(train_idf)

#predict the lable on the test data
predict_train = model_LR.predict(test_idf)

# f1 score on train data
f1_score(y_true= train.label, y_pred = predict_train)
print (f1_score)

f1_score(y_true= test.label, y_pred = predict_test)
print (f1_score)
