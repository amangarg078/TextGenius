from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import os
import pickle
import cPickle

def text_sentiment(docs_new):
   docs_new=[docs_new]
   twenty_train= load_files('.//Sentiment')  #the complete data is in this directory; like comp.graphics etc
   count_vect = CountVectorizer()
   X_train_counts = count_vect.fit_transform(twenty_train.data)
   tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
   X_train_tf = tf_transformer.transform(X_train_counts)
   tfidf_transformer = TfidfTransformer()
   X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

   # Fit a classifier on the training set
   #clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
   #f = open('my_classifier.pickle', 'wb')
   #pickle.dump(clf, f)
   #f = open('my_classifier.pickle',)
   #clf = pickle.load(f)
   #f.close()
   # save the classifier
   #with open('my_sentiment.pkl', 'wb') as fid:
      #cPickle.dump(clf, fid)    

   # load it again
   with open('my_sentiment.pkl', 'rb') as fid:
      clf = cPickle.load(fid)
   X_new_counts = count_vect.transform(docs_new)
   X_new_tfidf = tfidf_transformer.transform(X_new_counts)

   predicted = clf.predict(X_new_tfidf)
   return twenty_train.target_names[predicted]


