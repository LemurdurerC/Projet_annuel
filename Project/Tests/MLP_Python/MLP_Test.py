import ctypes
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path_to_dll = "../../Lib/MLPCppLib/cmake-build-debug/MLPCppLib.dll"

    my_lib = ctypes.CDLL(path_to_dll)

    my_lib.create_MLP_model.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    my_lib.create_MLP_model.restype = ctypes.c_void_p

    my_lib.dispose_MLP.argtypes = [ctypes.c_void_p]
    my_lib.dispose_MLP.restype = None

    my_lib.predict_MLP_Classification.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double)
    ]
    my_lib.predict_MLP_Classification.restype = ctypes.POINTER(ctypes.c_double)

    my_lib.predict_MLP_Regression.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double)
    ]
    my_lib.predict_MLP_Regression.restype = ctypes.POINTER(ctypes.c_double)

    my_lib.train_MLP_Classification.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int
    ]
    my_lib.train_MLP_Classification.restype = None

    my_lib.train_MLP_Regression.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int
    ]
    my_lib.train_MLP_Regression.restype = None
    """
    L = np.array([
        2,
        2,
        1
    ], dtype='int32')

    model = my_lib.create_MLP_model(L.shape[0], L.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))



    #CAS DE TEST : OU EXCLUSIF
    X = np.array([
        [1, 0],
        [0, 1],
        [0, 0],
        [1, 1]
    ], dtype='float64')

    Y = np.array([
        -1,
        1,
        1,
        -1
    ], dtype='float64')

    flattened_X = X.flatten()

    print("Before Training the Mannnn")
    for inputs_k in X:
        Exit1 = my_lib.predict_MLP_Classification(
            model,
            L.shape[0],
            L.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            inputs_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        print(Exit1[0])


    my_lib.train_MLP_Classification(
        model,
        L.shape[0],
        L.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        flattened_X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        X.shape[0],
        X.shape[1],
        0.1,
        100000
    )

    print("After Training the ...")
    for inputs_k in X:
        Exit2 = my_lib.predict_MLP_Classification(
            model,
            L.shape[0],
            L.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            inputs_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        print(Exit2[0])

    my_lib.dispose_MLP(model)

    """

# CAS DE TEST 1 : Linear Simple
    L = np.array([
        2,
        2,
        1
    ], dtype='int32')

    model = my_lib.create_MLP_model(L.shape[0], L.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))

    X = np.array([
        [1, 0],
        [0, 1],
        [4, 7],
        [11, 9]
    ], dtype='float64')

    Y = np.array([
        1,
        1,
        -1,
        -1
    ], dtype='float64')

    flattened_X = X.flatten()

    print("Before Training the Mannnn")
    for inputs_k in X:
        Exit1 = my_lib.predict_MLP_Classification(
            model,
            L.shape[0],
            L.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            inputs_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        print(Exit1[0])

    my_lib.train_MLP_Classification(
        model,
        L.shape[0],
        L.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        flattened_X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        X.shape[0],
        X.shape[1],
        0.01,
        10000
    )

    print("After Training the ...")
    for inputs_k in X:
        Exit2 = my_lib.predict_MLP_Classification(
            model,
            L.shape[0],
            L.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            inputs_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        print(Exit2[0])


    my_lib.dispose_MLP(model)
    
