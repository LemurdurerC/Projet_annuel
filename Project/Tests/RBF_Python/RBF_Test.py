import ctypes
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path_to_dll = "../../Lib/RBFLib/cmake-build-debug/RBFLib.dll"

    my_lib = ctypes.CDLL(path_to_dll)

    my_lib.create_RBF_model.argtypes = [ctypes.c_int]
    my_lib.create_RBF_model.restype = ctypes.c_void_p

    my_lib.disposeRBF.argtypes = [ctypes.c_void_p,
                                  ctypes.c_void_p]
    my_lib.disposeRBF.restype = None

    my_lib.KMeans.argtypes = [ctypes.c_int,
                              ctypes.POINTER(ctypes.c_double),
                              ctypes.c_int,
                              ctypes.c_int]
    my_lib.KMeans.restype = ctypes.c_void_p

    my_lib.RBF_predict_model_Regression.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        #ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int
    ]
    my_lib.RBF_predict_model_Regression.restype = ctypes.c_double

    my_lib.RBF_predict_model_Classification.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int
    ]
    my_lib.RBF_predict_model_Classification.restype = ctypes.c_double

    my_lib.RBF_train_model.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double
    ]
    my_lib.RBF_train_model.restype = None


    X = np.array([
        [1.0, 3.0],
        [1.0, 4.0],
        [2.0, 2.0],
        [2.0, 5.0],
        [3.0, 3.0],

        [6.0, 7.0],
        [7.0, 8.0],
        [8.0, 9.0],
        [8.0, 5.0],
        [9.0, 8.0]

    ], dtype='float64')

    Y = np.array([
        -1,
        -1,
        -1,
        -1,
        -1,

        1,
        1,
        1,
        1,
        1
    ], dtype='float64')

    flattened_X = X.flatten()

    numberOfCluser = 2
    gamma = 0.01

    KM = my_lib.KMeans(numberOfCluser, flattened_X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), X.shape[0], X.shape[1])
    model = my_lib.create_RBF_model(ctypes.c_int(numberOfCluser))


    print("Before Training the Modelco")
    for inputs_k in X:
        print(my_lib.RBF_predict_model_Classification(
            model,
            KM,
            numberOfCluser,
            gamma,
            inputs_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(inputs_k)
        ))
    my_lib.RBF_train_model(
        model,
        KM,
        numberOfCluser,
        flattened_X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        X.shape[0],
        X.shape[1],
        gamma
    )

    print("After Training the Model")
    for inputs_k in X:
        print(my_lib.RBF_predict_model_Classification(
            model,
            KM,
            numberOfCluser,
            gamma,
            inputs_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(inputs_k)
        ))


my_lib.disposeRBF(model,KM)
