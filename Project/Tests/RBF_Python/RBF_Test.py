import ctypes
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def percentOfGoodPrediction(all,part):
    return 100 - ((100*part)/all)


def percentOfBadPrediction(all,part):
    return (100*part)/all


if __name__ == "__main__":
    path_to_dll = "../../Lib/RBFLib/cmake-build-debug/RBFLib.dll"

    my_lib = ctypes.CDLL(path_to_dll)

    my_lib.create_RBF_model.argtypes = [ctypes.c_int]
    my_lib.create_RBF_model.restype = ctypes.c_void_p

    my_lib.disposeRBF.argtypes = [ctypes.c_void_p,
                                  ctypes.POINTER(ctypes.c_double)]
                                  #ctypes.c_void_p]
    my_lib.disposeRBF.restype = None

    my_lib.KMeans.argtypes = [ctypes.c_int,
                              ctypes.POINTER(ctypes.c_double),
                              ctypes.c_int,
                              ctypes.c_int,
                              ctypes.c_int,
                              ctypes.c_int]
    my_lib.KMeans.restype = ctypes.POINTER(ctypes.c_double)


    my_lib.RBF_predict_model_Regression.argtypes = [
        ctypes.c_void_p,
        #ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int
    ]
    my_lib.RBF_predict_model_Regression.restype = ctypes.c_double

    my_lib.RBF_predict_model_Classification.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        #ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int
    ]
    my_lib.RBF_predict_model_Classification.restype = ctypes.c_double

    my_lib.RBF_train_model.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        #ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double
    ]
    my_lib.RBF_train_model.restype = None



    #CAS OU CA MARCHE BIEN
    # gamma = 0.01
    # nnumbeCluster = 2
    # min = 0
    # max = 9
    A = np.array([
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

    B = np.array([
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





    # CLASSIFICATION

    # CAS DE TEST 1 : Linear simple
    # gamma = 0.1
    # numberCluster = 2
    # min = 0
    # max = 3

    C = np.array([
        [1, 1],
        [2, 3],
        [3, 3]
    ], dtype='float64')
    D = np.array([
        1,
        -1,
        -1
    ], dtype='float64')





    # CAS DE TEST 2 : Linear multiple
    # gamma = 0.001
    # numberCluster = 2
    # min = 0
    # max = 3
    E = np.concatenate(
        [np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
    Ftemp = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])

    F = Ftemp.flatten()





    # CAS DE TEST 3 : XOR
    # gamma = 0.01
    # numberCluster = 2
    # min = 0
    # max = 1

    G = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    H = np.array([1, 1, -1, -1])





    # CAS DE TEST 4 : CROSS (not work)
    # gamma = 0.01
    # nnumbeCluster = 2
    # min = -9
    # max = 9
    I = np.random.random((500, 2)) * 2.0 - 1.0
    J = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in I])





    # REGRESSION

    # CAS DE TEST 1 : Linear Simple 2D (not work)
    # gamma = 0.01
    # nnumbeCluster = 2
    # min = 0
    # max = 2
    K = np.array([
        [1],
        [2]
    ])
    L = np.array([
        2,
        3
    ])





    # CAS DE TEST 2 : Non linear simple 2D (not work)
    # gamma = 0.01
    # nnumbeCluster = 2
    # min = 1
    # max = 3

    M = np.array([
        [1],
        [2],
        [3]
    ])
    N = np.array([
        2,
        3,
        2.5
    ])





    # CAS DE TEST 3: Linear simple 3D (not work)
    # gamma = 0.1
    # nnumbeCluster = 2
    # min = 1
    # max = 3

    O = np.array([
        [1, 1],
        [2, 2],
        [3, 1]
    ])
    P = np.array([
        2,
        3,
        2.5
    ])





    min = 0
    max = 9
    numberOfCluser = 2
    gamma = 0.01
    enter = A
    exit = B
    flattened_X = enter.flatten()


    KM = my_lib.KMeans(numberOfCluser, flattened_X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), enter.shape[0],
                       enter.shape[1], min, max)


    model = my_lib.create_RBF_model(ctypes.c_int(numberOfCluser))


    for i in range(numberOfCluser*2):
        print(KM[i])


    my_lib.RBF_train_model(
        model,
        KM,
        numberOfCluser,
        flattened_X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        exit.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        enter.shape[0],
        enter.shape[1],
        gamma
    )

    print("After Training the Model")
    count = 0
    bad = 0
    error = 0.2

    for inputs_k in enter:
        result = (my_lib.RBF_predict_model_Classification(
            model,
            KM,
            numberOfCluser,
            gamma,
            inputs_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(inputs_k)
        ))
        print(result)
        if result != exit[count]:
            if abs(result - exit[count]) > error:
                bad = bad + 1
        count = count + 1

    print("Pourcentage")
    print(percentOfGoodPrediction(enter.shape[0], bad), "% de bonne prédiction")
    print(percentOfBadPrediction(enter.shape[0], bad), "% de mauvaise prédiction")

my_lib.disposeRBF(model,KM)





