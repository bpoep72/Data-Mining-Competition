
import scipy.io
import numpy

output = numpy.load("Predictions-randomForest-500-50-40.npy")

output_dictionary = {"predictions_500_50_40" : output}

output_filename = "prediction-rf-500-50-40"

scipy.io.savemat(output_filename, output_dictionary)