
import scipy
import scipy.io
import os
import numpy

class Data:

    def __init__(self):

        vocabulary = None
        testing_data = None
        training_data = None
        training_labels = None

        self.get_data()
    
    def get_data(self):

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'vocabulary.mat')

        self.vocabulary = scipy.io.loadmat(path)['vocab']
        self.vocabulary = numpy.squeeze(self.vocabulary)

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'testing.mat')

        self.testing_data = scipy.io.loadmat(path)['Xt']

        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'training.mat')

        self.training_data = scipy.io.loadmat(path)['Xn']

        self.training_labels = scipy.io.loadmat(path)['Yn']
