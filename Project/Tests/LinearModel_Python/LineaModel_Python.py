import ctypes
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def percentOfGoodPrediction(all,part):
    return 100 - ((100*part)/all)


def percentOfBadPrediction(all,part):
    return (100*part)/all


if __name__ == "__main__":

    path_to_dll = "../../Lib/LinearModelCppLib/cmake-build-debug/LinearModelCppLib.dll"

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

    my_lib.ecriture.argtypes = [ctypes.c_void_p,
                                ctypes.c_int]
    my_lib.ecriture.restype = None

    my_lib.lecture.argtypes = [ctypes.c_int]
    my_lib.lecture.restype = ctypes.POINTER(ctypes.c_double)

    #CLASSIFICATION

    # CAS DE TEST 1 : Linear Simple
    # 0.01
    # 1000
    A = np.array([
        [1, 1],
        [2, 3],
        [3, 3]
    ], dtype='float64')

    B = np.array([
        1,
        -1,
        -1
    ], dtype='float64')




    #CAS DE TEST : LINEAR MULTIPLE
    # 0.01
    # 1000
    C = np.concatenate(
        [np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
    D = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])
    DFlat = D.flatten()




    #CAS DE TEST : OU EXCLUSIF
    # 0.1
    # 100
    E = np.array([
        [1, 0],
        [0, 1],
        [0, 0],
        [1, 1]
    ], dtype='float64')

    F = np.array([
        1,
        1,
        -1,
        -1
    ], dtype='float64')




    #CAS DE TEST : CROSS
    # 1
    # 10000
    G = np.random.random((500, 2)) * 2.0 - 1.0
    H = np.array([1.0 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1.0 for p in G])





    # REGRESSION

    # CAS DE TEST : Linear Simple 2D
    # 0.1
    # 10000
    I = np.array([
        [1],
        [2]
    ], dtype='float64')
    J = np.array([
        2,
        3
    ], dtype='float64')





    # CAS DE TEST : Non linear simple 2D
    # 0.1
    # 500000
    M = np.array([
        [1],
        [2],
        [3]
    ], dtype='float64')
    N = np.array([
        2,
        3,
        2.5
    ], dtype='float64')




    # CAS DE TEST : Linear simple 3D
    # 0.01
    # 500000
    O = np.array([
        [1, 1],
        [2, 2],
        [3, 1]
    ], dtype='float64')
    P = np.array([
        2,
        3,
        2.5
    ], dtype='float64')





    # CAS DE TEST : Linear Tricky 3D
    # 0.01
    # 10000
    Q = np.array([
        [1, 1],
        [2, 2],
        [3, 3]
    ], dtype='float64')
    R = np.array([
        1,
        2,
        3
    ], dtype='float64')





    # CAS DE TEST : Non linear simple 3D
    # 1 ?
    # 100000 ?
    S = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
    ], dtype='float64')
    T = np.array([
        2,
        1,
        -2,
        -1
    ], dtype='float64')




    enter = O
    exit = P
    alpha = 0.1
    iteration = 1000
    flattened_X = enter.flatten()

    model = my_lib.linear_create_model(ctypes.c_int(enter.shape[1]))

    my_lib.linear_train_model_classification(
        model,
        flattened_X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        exit.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        enter.shape[0],
        enter.shape[1],
        #exit.shape[0]
        alpha,
        iteration
    )

    my_lib.ecriture(model, ctypes.c_int(enter.shape[1]))
    
    print("After Training...")
    count = 0
    bad = 0
    error = 0.3
    for inputs_k in enter:
        result = my_lib.linear_predict_model_classification(
            model,
            inputs_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(inputs_k)
        )
        print(result)
        if result != exit[count]:
            if abs(result - exit[count]) > error:
                bad = bad + 1
        count = count + 1

    print("Pourcentage")
    print(percentOfGoodPrediction(enter.shape[0], bad), "% de bonne prédiction")
    print(percentOfBadPrediction(enter.shape[0], bad), "% de mauvaise prédiction")

    my_lib.linear_dispose_model(model)















