# Libraries Used
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import itertools
import facebook
from time import sleep
classes = ["POSITIVE", "NEGATIVE"]


# Auxilary Functions
def cnf_plot(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Loading files

training_file = 'C:\\training.csv'
test_file = 'C:\\test.csv'
df_train = pd.read_csv(training_file, header=None, usecols=[0, 5], index_col=None, encoding="iso-8859-1")
label_train = df_train[0].as_matrix()
label_train = np.concatenate((label_train[:100000], label_train[800000:900000]))
tweets_train = df_train[5].as_matrix()
tweets_train = np.concatenate((tweets_train[:100000], tweets_train[800000:900000]))
df_test = pd.read_csv(test_file, header=None, usecols=[0, 5], index_col=None, encoding="iso-8859-1")
tweets_test = df_test[5].as_matrix()
label_test = df_test[0].as_matrix()

# Preprocessor

stop = list(stopwords.words('english'))
linkwords = ["www", "http", "https", "com"]
stop += linkwords
tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer("english")


def customTokens(s):
    word = tokenizer.tokenize(s)
    a = []
    for w in word:
        if not w.isdigit():
            a.append(stemmer.stem(w))
    return a


# End of preprocessing


# Feature Extraction

vectorizer = CountVectorizer(analyzer='word', strip_accents='ascii', stop_words=stop, tokenizer=customTokens, min_df=2)
x_train = vectorizer.fit_transform(tweets_train)
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train)
remove2 = label_test != 2
sent_test = label_test[remove2]
tweets_test = tweets_test[remove2]
# Use them if feature Selection is not required
# x_test = vectorizer.transform(tweets_test)
# x_test_tfidf = tfidf_transformer.transform(x_test)


# Feature Selection

ch2 = SelectKBest(chi2, k=1000)
new_train = ch2.fit_transform(x_train, label_train)
tfidf_transformer = TfidfTransformer()
x_train_tfidf_n = tfidf_transformer.fit_transform(new_train)
# vocabulary=np.asarray(vectorizer.get_feature_names())[ch2.get_support()]
# vectorizer2 = CountVectorizer(analyzer='word', strip_accents='ascii', stop_words=stop, tokenizer=customTokens,vocabulary=vocabulary)
x_test = vectorizer.transform(tweets_test)
x_test = ch2.transform(x_test)
x_test_tfidf_n = tfidf_transformer.transform(x_test)

# Classifier Phase

# SVM

svm_classifier_tfidf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)
svm_classifier_tfidf.fit(x_train_tfidf_n, label_train)
print(svm_classifier_tfidf.score(x_test_tfidf_n, sent_test))
result = svm_classifier_tfidf.predict(x_test_tfidf_n)
cnf=confusion_matrix(sent_test,result)
print(cnf)
fig = plt.figure()
fig.canvas.set_window_title('State Vector Machine')
cnf_plot(cnf)
plt.show()

print(classification_report(sent_test,result,target_names=["POSITIVE","NEGATIVE"]))
svm_classifier_tf=SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)
svm_classifier_tf.fit(new_train, label_train)
print(svm_classifier_tf.score(x_test,sent_test))

#KNN

knn_classifier_tfidf = KNeighborsClassifier(n_neighbors=5)
knn_classifier_tfidf.fit(x_train_tfidf_n, label_train)
print(knn_classifier_tfidf.score(x_test_tfidf_n,sent_test))
result=svm_classifier_tfidf.predict(x_test_tfidf_n)
cnf=confusion_matrix(sent_test,result)
print(cnf)
fig = plt.figure()
fig.canvas.set_window_title('K Nearest Neighbour')
cnf_plot(cnf)
plt.show()
print(classification_report(sent_test,result,target_names=["POSITIVE","NEGATIVE"]))

knn_classifier_tf=KNeighborsClassifier(n_neighbors=5)
knn_classifier_tf.fit(new_train,label_train)
print(knn_classifier_tf.score(x_test,sent_test))
result_knn_tf=knn_classifier_tf.predict(x_test)
print(confusion_matrix(sent_test,result_knn_tf))

#NB

NB=MultinomialNB()
NB.fit(new_train,label_train)
print(NB.score(x_test,sent_test))
NB_tfidf=MultinomialNB()
NB_tfidf.fit(x_train_tfidf_n, label_train)
print(NB_tfidf.score(x_test_tfidf_n,sent_test))
result=svm_classifier_tfidf.predict(x_test_tfidf_n)
cnf=confusion_matrix(sent_test,result)
print(cnf)
fig = plt.figure()
fig.canvas.set_window_title('Naive Bayes')
cnf_plot(cnf)
plt.show()
print(classification_report(sent_test,result,target_names=["POSITIVE","NEGATIVE"]))


# FACEBOOK API

token = "EAACEdEose0cBAAzUfJcyjuKvIZBzc4XMZAQ3vY9M24RmZB1YIGcq3ASArZBEGdaZAsnHYmZB9fZCWhp69z75AsH5ZCLQc4DLFxYVwklbehYc2cJym7CGYh0tbb2ZCu7wVOLhW1S2PAyhaEsG73MNVyKrnlMLcZBsccC1ZBeLrjKyICziwaBZBoYKyRmpHcNXGH0T0BkZD"
graph = facebook.GraphAPI(access_token=token)
profile = graph.get_object('TheWalkingDeadGame')
posts = graph.get_connections(profile['id'], 'posts')
reaction = []
index = []
i = 0
best_post = ""
for post in posts['data']:
    if i > 12:
        break
    positives = 0
    negatives = 0
    comments = graph.get_connections(post['id'], 'comments')
    j = 0

    for comment in comments['data']:
        if j > 50:
            break;
        test = vectorizer.transform([comment['message']])
        test = ch2.transform(test)
        test_tfidf_n = tfidf_transformer.transform(test)
        x = svm_classifier_tfidf.predict(test_tfidf_n)
        if x == 0:
            positives = positives + 1
        else:
            negatives = negatives + 1
    if positives == 0 and negatives == 0:
        reaction.append(0)
    else:
        reaction.append(positives / (positives + negatives) * 100)
    index.append(i + 1)
    i = i + 1
    sleep(4)
    if  i != 1:
        if (reaction[i-1] > reaction[i - 2]):
         try:
                best_post = post["message"]
         except KeyError:
              continue

print(best_post)
fig = plt.figure()
fig.canvas.set_window_title('Facebook Posts Opinion')
plt.ylim([0, 100])
plt.xlabel('Posts')
plt.ylabel('Postitive Reaction(%)')
plt.plot(index, reaction)
plt.show()

# Statistics

# Accuracy in sequence (without feature selection)
# 0.743732590529
# 0.779944289694
# 0.529247910864
# 0.66852367688

# Accuracy in sequence (Feature Selection)
# 0.763231197772
# 0.74651810585
# 0.732590529248
# 0.710306406685


# 2000 feature selection
# 0.796657381616
# 0.777158774373
# 0.713091922006
# 0.710306406685
