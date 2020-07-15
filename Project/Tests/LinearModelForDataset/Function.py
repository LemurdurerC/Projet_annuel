import numpy as np
import ctypes
import ImageUpload


"""
def seeUnderFeating(Dataset,DatasetExpectedOutput):
    print("Détection sous apprentissage:")
    k = 0
    total = 0
    for inputs_k in Dataset:
        output = my_lib.linear_predict_model_classification(
            model,
            inputs_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(inputs_k)
        )
        if output != DatasetExpectedOutput[k]:
            total = total + 1
        k = k + 1

    percent = 100 - ((100 * total) / dataset.shape[0])
    print(percent, " %")

"""