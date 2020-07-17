import ctypes
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def percentOfGoodPrediction(all,part):
    return 100 - ((100*part)/all)


def percentOfBadPrediction(all,part):
    return (100*part)/all


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



    #CLASSIFICATION

    # CAS DE TEST 1 : Linear Simple
    # 0.01
    # 50000
    # 2 1
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
    # 0.1
    # 100000
    # 2 1
    C = np.concatenate(
        [np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
    D = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])
    DFlat = D.flatten()




    #CAS DE TEST : OU EXCLUSIF
    # 0.1
    # 100000
    # 2 2 1
    E = np.array([
        [1, 0],
        [0, 1],
        [0, 0],
        [1, 1]
    ], dtype='float64')

    F = np.array([
        -1,
        1,
        1,
        -1
    ], dtype='float64')




    #CAS DE TEST : CROSS
    # 1
    # 10000
    # 2 4 1
    G = np.random.random((500, 2)) * 2.0 - 1.0
    H = np.array([1.0 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1.0 for p in G])




    #REGRESSION

    #CAS DE TEST : Linear Simple 2D
    # 0.1
    # 10000
    # 1 1
    I = np.array([
        [1],
        [2]
    ], dtype='float64')
    J = np.array([
        2,
        3
    ], dtype='float64')




    #CAS DE TEST : Non linear simple 2D
    # 0.1
    # 500000
    # 1 3 1
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




    #CAS DE TEST : Linear simple 3D
    # 0.01
    # 500000
    # 2 1
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




    #CAS DE TEST : Linear Tricky 3D
    # 0.01
    # 10000
    # 2 1
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





    #CAS DE TEST : Non linear simple 3D
    # 1 ?
    # 100000 ?
    # 2 2 1
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





    L = np.array([
        2,
        2,
        1
    ], dtype='int32')

    model = my_lib.create_MLP_model(L.shape[0], L.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))

    enter = S
    exit = T
    alpha = 1
    iteration = 100000

    flattened_X = enter.flatten()

    my_lib.train_MLP_Regression(
        model,
        L.shape[0],
        L.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        flattened_X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        exit.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        enter.shape[0],
        enter.shape[1],
        alpha,
        iteration
    )


    print("After Training...")
    count = 0
    bad = 0
    error = 0.2
    for inputs_k in enter:
        result = my_lib.predict_MLP_Regression(
            model,
            L.shape[0],
            L.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            inputs_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
        print(result[0])
        if result[0] != exit[count]:
            if abs(result[0] - exit[count]) > error:
                bad = bad + 1
            count = count + 1

    print("Pourcentage")
    print(percentOfGoodPrediction(enter.shape[0], bad), "% de bonne prédiction")
    print(percentOfBadPrediction(enter.shape[0], bad), "% de mauvaise prédiction")


my_lib.dispose_MLP(model)
    
