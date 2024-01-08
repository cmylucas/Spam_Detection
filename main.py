# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import string
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import pickle

nltk.download('stopwords')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('./'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


flist = glob.glob('mail/*.txt')
mails = []
for file in flist:
    spm = 0
    if file[5:8] == 'spm':
        spm = 1
    with open(file, 'r') as f:
        mails.append([f.read(),spm])


mails = pd.DataFrame(mails, columns = ['text', 'spam'])

def plot_word_cloud(data, typ):
    email_corpus = " ".join(data['text'])
    
    plt.figure(figsize=(7, 7))
    
    wc = WordCloud(background_color='black',
                   max_words=100,
                   width=800,
                   height=400,
                   collocations=False).generate(email_corpus)
    # print(wc.process_text(email_corpus))
    plt.imshow(wc, interpolation='bilinear')
    plt.title(f'WordCloud for {typ} emails', fontsize=15)
    plt.axis('off')
    plt.show()

def linkcounter(mails):
    spam_mails = mails[mails['spam']==1]
    non_spam_mails = mails[mails['spam']==0]

    spam_with_links = (spam_mails['text'].str.contains('http')).mean()
    non_spam_with_links = (non_spam_mails['text'].str.contains('http')).mean()
    
    categories = ['Spam', 'Non-Spam']
    proportions = [spam_with_links, non_spam_with_links]

    plt.figure(figsize=(7, 7))
    plt.bar(categories, proportions)
    plt.xlabel('Email Category')
    plt.ylabel('Proportion of Emails with Links')
    plt.title('Proportion of Emails with Links in Spam and Non-Spam Categories')
    plt.show()

plot_word_cloud(mails[mails['spam'] == 0], typ='Non-Spam')
plot_word_cloud(mails[mails['spam'] == 1], typ='Spam')

linkcounter(mails)

# c=0
# x=0
# while (c <10):
#     # print(x)
#     if mails.loc[x]['spam'] == 1:
#         print("Real")
#         print(mails.loc[c]['text'])
#         print("Spam")
#         print(mails.loc[x]['text'])
#         c += 1
#     x += 1

def remove_stopwords(text):
    stop_words = stopwords.words('english')
    imp_words = []
    for word in str(text).split():
        word = word.lower()
 
        if word not in stop_words:
            imp_words.append(word)
 
    output = " ".join(imp_words)
 
    return output
def remove_punctuations(text):
    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)

mails['text'] = mails['text'].str.replace('Subject', '')
punctuations_list = string.punctuation
mails['text']= mails['text'].apply(lambda x: remove_punctuations(x))
mails['text'] = mails['text'].apply(lambda text: remove_stopwords(text))

# c=0
# x=0
# while (c <10):
#     # print(x)
#     if mails.loc[x]['spam'] == 1:
#         print("Real")
#         print(mails.loc[c]['text'])
#         print("Spam")
#         print(mails.loc[x]['text'])
#         c += 1
#     x += 1


cv = CountVectorizer(decode_error='ignore')
X = cv.fit_transform(mails['text'])
with open('count_vectorizer.pkl', 'wb') as f:
    pickle.dump(cv, f)
# print(X)
X_train, X_test, y_train, y_test = train_test_split(X, mails['spam'], test_size=0.3, random_state=69)

#Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train,y_train)
# print(mnb.predict(X_test))
# print(y_test)
print('Naive Bayes training accuracy: ',mnb.score(X_train,y_train)*100)
print('Naive Bayes test accuracy: ',mnb.score(X_test,y_test)*100)

#Multi-Layer Perceptron
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=101)
mlp.fit(X_train, y_train)
# y_pred = mlp.predict(X_test)
# print(y_pred)
print('Multi-Layer Perceptron training accuracy: ',mlp.score(X_train,y_train)*100)
print('Multi-Layer Perceptron test accuracy: ',mlp.score(X_test,y_test)*100)



layers = [(25,25), (25,50), (50, 50), (100, 50)]
max_it = [100,200,300]
max = 0
para = []
model = None
for l in layers:
    for m in max_it:
        print(l,m)
        mlp = MLPClassifier(hidden_layer_sizes=l, max_iter=m, random_state=101)
        mlp.fit(X_train, y_train)
        score = mlp.score(X_test,y_test)*100
        if score >= max:
            max = score
            para = [l,m]
            model = mlp
            print(model)

print(max)
print(para)
with open('model.pkl', 'wb') as f:
    pickle.dump(model,f)


# print(X_test)
def spamPredict(email):
    if not type(email) == str:
        return None
    model = None
    cv=None
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('count_vectorizer.pkl', 'rb') as f:
        cv = pickle.load(f)
    data = np.array([[email]])
    dataframe = pd.DataFrame(data,columns = ['text'])

    dataframe['text'] = dataframe['text'].str.replace('Subject', '')
    punctuations_list = string.punctuation
    dataframe['text']= dataframe['text'].apply(lambda x: remove_punctuations(x))
    dataframe['text'] = dataframe['text'].apply(lambda text: remove_stopwords(text))
    
    # print(dataframe)
    X = cv.transform(dataframe['text'])
    # print(X)
    y_pred = model.predict(X)
    return y_pred[0]

print(spamPredict("this is a spam email lol believe"))
