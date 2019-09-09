
import scipy.io
import numpy

output = numpy.load("Predictions-SVC-false-01.npy")

output_dictionary = {"predictions_svc_0_1" : output}

output_filename = "predictions_svc_0_1"

scipy.io.savemat(output_filename, output_dictionary)