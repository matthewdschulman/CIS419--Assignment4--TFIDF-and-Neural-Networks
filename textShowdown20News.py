'''
    Text Classification and ROC
    AUTHOR Matt Schulman
'''

import numpy as np

from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# import the 20 newsgroups data set (training)
newsgroups_train = fetch_20newsgroups(subset='train')

# import the 20 newsgroups data set (testing)
newsgroups_test = fetch_20newsgroups(subset='test')

text_clf = Pipeline([('vect', CountVectorizer(lowercase=True, stop_words='english')),
		     ('tfidf', TfidfTransformer(norm='l2', sublinear_tf=True)),
		     ('clf', MultinomialNB()),
		   ])

text_clf = text_clf.fit(newsgroups_train.data, newsgroups_train.target)

docs_test = newsgroups_test.data
predicted = text_clf.predict(docs_test)
print(np.mean(predicted == newsgroups_test.target))
print(metrics.classification_report(newsgroups_test.target, predicted, target_names=newsgroups_test.target_names))
