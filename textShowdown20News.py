'''
    Text Classification and ROC
    AUTHOR Matt Schulman
'''

import numpy as np
import time

from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from tabulate import tabulate
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm

# import the 20 newsgroups data set (training)
newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True, remove=('headers', 'footers'))

# import the 20 newsgroups data set (testing)
newsgroups_test = fetch_20newsgroups(subset='test', shuffle=True, remove=('headers', 'footers'))

text_clf_bayes = Pipeline([('vect', CountVectorizer(lowercase=True, stop_words='english')),
		     ('tfidf', TfidfTransformer(norm='l2', sublinear_tf=True)),
		     ('clf', MultinomialNB()),
		   ])
time_before_nb_training = time.clock()
text_clf_bayes = text_clf_bayes.fit(newsgroups_train.data, newsgroups_train.target)
time_after_nb_training = time.clock()

text_clf_svm = Pipeline([('vect', CountVectorizer(lowercase=True, stop_words='english')),
			 ('tfidf', TfidfTransformer(norm='l2', sublinear_tf=True)),
			 ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),
		       ])

time_before_svm_training = time.clock()
text_clf_svm = text_clf_svm.fit(newsgroups_train.data, newsgroups_train.target)
time_after_svm_training = time.clock()

print "NB..."
predicted_train = text_clf_bayes.predict(newsgroups_train.data)
predicted_test = text_clf_bayes.predict(newsgroups_test.data)

accuracy_train = np.mean(predicted_train == newsgroups_train.target)
accuracy_test = np.mean(predicted_test == newsgroups_test.target)
stats_train = precision_recall_fscore_support(newsgroups_train.target, predicted_train, average='binary')
stats_test = precision_recall_fscore_support(newsgroups_test.target, predicted_test, average='binary')
training_time = time_after_nb_training - time_before_nb_training

nb_table = [['STATISTIC', 'TRAINING', 'TESTING'],
	    ['accuracy', accuracy_train, accuracy_test],
	    ['precision', stats_train[0], stats_test[0]],
	    ['recall', stats_train[1], stats_test[1]],
	    ['training time (seconds)', training_time, 'N/A']
	    ]
print tabulate(nb_table)


print "SVM..."
predicted_train = text_clf_svm.predict(newsgroups_train.data)
predicted_test = text_clf_svm.predict(newsgroups_test.data)

accuracy_train = np.mean(predicted_train == newsgroups_train.target)
accuracy_test = np.mean(predicted_test == newsgroups_test.target)
stats_train = precision_recall_fscore_support(newsgroups_train.target, predicted_train, average='binary')
stats_test = precision_recall_fscore_support(newsgroups_test.target, predicted_test, average='binary')
training_time = time_after_svm_training - time_before_svm_training

nb_table = [['STATISTIC', 'TRAINING', 'TESTING'],
	    ['accuracy', accuracy_train, accuracy_test],
	    ['precision', stats_train[0], stats_test[0]],
	    ['recall', stats_train[1], stats_test[1]],
	    ['training time (seconds)', training_time, 'N/A']
	    ]
print tabulate(nb_table)
