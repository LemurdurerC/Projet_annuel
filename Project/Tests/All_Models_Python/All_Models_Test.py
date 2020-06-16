import ctypes
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == "__main__":

    path_to_dll = "../LinearModelCppLib/cmake-build-debug" \
                  "/LinearModelCppLib.dll "

    my_lib = ctypes.CDLL(path_to_dll)

    my_lib.linear_create_model.argtypes = [ctypes.c_int]
    my_lib.linear_create_model.restype = ctypes.c_void_p

    my_lib.linear_dispose_model.argtypes = [ctypes.c_void_p]
    my_lib.linear_dispose_model.restype = None

    my_lib.linear_train_model_classification.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int
    ]
    my_lib.linear_train_model_classification.restype = None

    my_lib.linear_train_model_regression.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
    ]
    my_lib.linear_train_model_regression.restype = None

    my_lib.linear_predict_model_classification.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int
    ]
    my_lib.linear_predict_model_classification.restype = ctypes.c_double

    my_lib.linear_predict_model_regression.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int
    ]
    my_lib.linear_predict_model_regression.restype = ctypes.c_double

    model = my_lib.linear_create_model(ctypes.c_int(2))
 #   model = my_lib.linear_create_model(ctypes.c_int(7164160))

    #CLASSIFICATION
    X = np.array([
        [1, 1],
        [2, 3],
        [3, 3]
    ], dtype='float64')

    Y = np.array([
        1,
        -1,
        -1
    ], dtype='float64')

#REGRESSION
    K = np.array([
            [1, 1],
            [2, 2],
            [3, 1]
    ], dtype='float64')

    L = np.array([
        2,
        3,
        2.5
    ], dtype='float64')

    flattened_X = K.flatten()

    print("Before Training the Modelco")
    for inputs_k in K:
        print(my_lib.linear_predict_model_regression(
            model,
            inputs_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(inputs_k)
        ))

    my_lib.linear_train_model_regression(
        model,
        flattened_X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        L.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        K.shape[0],
        K.shape[1],
        L.shape[0]
        #0.01,
        #1000
    )

    print("After Training the Model")
    for inputs_k in K:
        print(my_lib.linear_predict_model_regression(
            model,
            inputs_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(inputs_k)
        ))

    my_lib.linear_dispose_model(model)















