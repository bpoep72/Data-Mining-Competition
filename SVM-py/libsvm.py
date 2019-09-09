
from sklearn.svm import LinearSVC

import numpy
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse.csr import csr_matrix

from dataimporter import Data

#get the data
data = Data()

#get counts of the data
testing_samples = numpy.size(data.testing_data, axis=0)
training_samples = numpy.size(data.training_data, axis=0)
labels = numpy.size(data.training_labels, axis=1)

""" #tfidf
tfidf = TfidfTransformer()

transformer = tfidf.fit(data.training_data)
#get the individual scores
scores = transformer.transform(data.training_data)

#get the scores across columns
scores = numpy.sum(scores, axis=0)
scores_sorted = numpy.sort(scores)
scores_sorted = numpy.fliplr(scores_sorted)

terms = 3000

best_scores = scores_sorted[:, 0:terms]

worst_of_the_best = best_scores[:, best_scores.shape[1] - 1]

#reduce the matrix based on the results of the tfidf
keep = scores > worst_of_the_best

keep_ind = []

for i in range(keep.shape[1]):
    if keep[0, i]:
        keep_ind.append(i)

keep_ind = numpy.asarray(keep_ind)

data.training_data = data.training_data[:, keep_ind]
data.testing_data = data.testing_data[:, keep_ind]
"""
predictions = numpy.zeros((testing_samples, labels))

for i in range(labels):
    
    model = LinearSVC(dual=False, C=.1)

    labels = csr_matrix(data.training_labels[:, i]).todense()

    model.fit(data.training_data, numpy.ravel(labels))

    predictions[:, i] = model.predict(data.testing_data)

    print("For " + str(i) + ": " + str(numpy.sum(predictions[:, i] == 1)))

    numpy.save('Predictions-SVC-hinge-false-01-balanced', predictions)


print("!!!!!!!!!!  I'm done, please put me to sleep now !!!!!!!!!!!!")
