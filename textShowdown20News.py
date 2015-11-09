'''
    Text Classification and ROC
    AUTHOR Matt Schulman
'''

import numpy as np
import time
import sklearn.metrics

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

# Naive Bayes Classifier and fitting
text_clf_bayes = Pipeline([('vect', CountVectorizer(lowercase=True, stop_words='english')),
                           ('tfidf', TfidfTransformer(norm='l2', sublinear_tf=True)),
		           ('clf', MultinomialNB()),
		          ]
			 )
time_before_nb_training = time.clock()
text_clf_bayes = text_clf_bayes.fit(newsgroups_train.data, newsgroups_train.target)
time_after_nb_training = time.clock()

# SVM Cosine Similiary Classifier and Fitting
# Note: if tfidftransformer data are normalized, cosine_similarity is equivalent to linear_kernel only
# slower, so this code uses the linear_kernel 
text_clf_svm = Pipeline([('vect', CountVectorizer(lowercase=True, stop_words='english')),
			 ('tfidf', TfidfTransformer(norm='l2', sublinear_tf=True)),
			 ('clf', svm.SVC(kernel=sklearn.metrics.pairwise.linear_kernel, probability=True)), 
		        ]
		       )

time_before_svm_training = time.clock()
text_clf_svm = text_clf_svm.fit(newsgroups_train.data, newsgroups_train.target)
time_after_svm_training = time.clock()

# Compute Naive Bayes predictions
print "NB..."
predicted_train_nb = text_clf_bayes.predict(newsgroups_train.data)
predicted_test_nb = text_clf_bayes.predict(newsgroups_test.data)

# Compute and print out Naive Bayes stats
accuracy_train_nb = np.mean(predicted_train_nb == newsgroups_train.target)
accuracy_test_nb = np.mean(predicted_test_nb == newsgroups_test.target)
stats_train_nb = precision_recall_fscore_support(newsgroups_train.target, predicted_train_nb, average='binary')
stats_test_nb = precision_recall_fscore_support(newsgroups_test.target, predicted_test_nb, average='binary')
training_time_nb = time_after_nb_training - time_before_nb_training

nb_table = [['STATISTIC', 'TRAINING', 'TESTING'],
	    ['accuracy', accuracy_train_nb, accuracy_test_nb],
	    ['precision', stats_train_nb[0], stats_test_nb[0]],
	    ['recall', stats_train_nb[1], stats_test_nb[1]],
	    ['training time (seconds)', training_time_nb, 'N/A']
	    ]
print tabulate(nb_table)

# Compute SVM + Cosine Similarity predictions
print "SVM..."
predicted_train_svm = text_clf_bayes.predict(newsgroups_train.data)
predicted_test_svm = text_clf_bayes.predict(newsgroups_test.data)

# Compute and print out SVM stats
accuracy_train_svm = np.mean(predicted_train_svm == newsgroups_train.target)
accuracy_test_svm = np.mean(predicted_test_svm == newsgroups_test.target)
stats_train_svm = precision_recall_fscore_support(newsgroups_train.target, predicted_train_svm, average='binary')
stats_test_svm = precision_recall_fscore_support(newsgroups_test.target, predicted_test_svm, average='binary')
training_time_svm = time_after_svm_training - time_before_svm_training

svm_table = [['STATISTIC', 'TRAINING', 'TESTING'],
	    ['accuracy', accuracy_train_svm, accuracy_test_svm],
	    ['precision', stats_train_svm[0], stats_test_svm[0]],
	    ['recall', stats_train_svm[1], stats_test_svm[1]],
	    ['training time (seconds)', training_time_svm, 'N/A']
	    ]
print tabulate(svm_table)

# Plot the ROC plots for both classifiers and the requested classes

