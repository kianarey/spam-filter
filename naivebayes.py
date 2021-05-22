import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix, precision_score,recall_score, f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#load in data
data = pd.read_csv('all_data.txt', header=None, keep_default_na=False)
data = data.to_numpy()
X = data[:,0].astype('str')
y = data[:,1].astype('int')

'''TfidfVectorizer and CountVectorizer both are methods for converting text data 
into vectors as model can process only numerical data. In CountVectorizer we 
only count the number of times a word appears in the document which results in 
biasing in favour of most frequent words. this ends up in ignoring rare words 
which could have helped is in processing our data more efficiently.

To overcome this , we use TfidfVectorizer. In TfidfVectorizer we consider 
overall document weightage of a word. It helps us in dealing with most frequent 
words. Using it we can penalize them. TfidfVectorizer weights the word counts 
by a measure of how often they appear in the documents.'''

def perform_CV(X):
    CV_vect = CountVectorizer() #Convert a collection of text documents to a matrix of token counts
    X_CV = CV_vect.fit(X)
    X_CV = CV_vect.transform(X).toarray()    
    return X_CV

def perform_TFIDF(X):
    TFIDF_vect = TfidfVectorizer() #Convert a collection of text documents to a matrix of token counts
    X_TFIDF = TFIDF_vect.fit(X)
    X_TFIDF = TFIDF_vect.transform(X).toarray()    
    return X_TFIDF

def naive_bayes(X, y, title): #Gaussian Naive Bayes
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    
    NB_clf = GaussianNB()
    NB_clf.fit(X_train, y_train) #train
    NB_train_score = NB_clf.score(X_train, y_train) #get scoring strength
    
    NB_test_score = NB_clf.score(X_test, y_test)
    NB_predictions = NB_clf.predict(X_test)
    
    #moment of truth
    tn, fp, fn, tp = confusion_matrix(y_test,NB_predictions).ravel()
    precision = precision_score(y_test, NB_predictions)
    recall = recall_score(y_test, NB_predictions)
    f1score = f1_score(y_test, NB_predictions)
    print(title)
    print("Precision: {:.2f}%".format(100 * precision))
    print("Recall: {:.2f}%".format(100 * recall))
    print("F1 Score: {:.2f}%".format(100 * f1score))
    
    #graph confusion matrix
    labels =['Non-Spam', 'Spam']
    fig = plot_confusion_matrix(NB_clf, X_test, y_test, cmap=plt.cm.Blues, display_labels=labels) 
    #fig.ax_.set_title('Confusion Matrix - %s' %(title))
    
    save(NB_train_score, NB_test_score, tn, fp, fn, tp, precision, recall, f1score, title)
    

def save(NB_train_score, NB_test_score, tn, fp, fn, tp, precision, recall, f1score, title):
    NB_save = {'Type': ['NB_train_score', 'NB_test_score', 'TN', 'FP', 'FN', 'TP', 'Precision', 'Recall', 'F1 Score'],
        'Value': [NB_train_score, NB_test_score, tn, fp, fn, tp, precision, recall, f1score]}
    NB_df = pd.DataFrame(NB_save, columns=['Type','Value'])
    NB_df.to_csv(title + '.txt', index=False)


X_CV = perform_CV(X)
naive_bayes(X_CV,y,title='Naive Bayes with CountVectorizer')

X_TFIDF = perform_TFIDF(X)
naive_bayes(X_TFIDF,y,title='Naive Bayes with TfidfVectorizer')

