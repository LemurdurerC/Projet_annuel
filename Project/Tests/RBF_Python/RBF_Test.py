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
"""
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

    KM = my_lib.KMeans(numberOfCluser, flattened_X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), X.shape[0],
                       X.shape[1], 0, 9)


    model = my_lib.create_RBF_model(ctypes.c_int(numberOfCluser))

    for i in range(numberOfCluser*2):
        print(KM[i])


    print("Before Training the Model")
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

    """









#CAS DE TEST 1 : Linear simple
"""
    
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

    flattened_A = A.flatten()

    numberOfCluser = 2
    gamma = 0.1

    KM = my_lib.KMeans(numberOfCluser, flattened_A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), A.shape[0],
                       A.shape[1], 0, 3)


    model = my_lib.create_RBF_model(ctypes.c_int(numberOfCluser))

    for i in range(numberOfCluser*2):
        print("next")
        print(KM[i])


    print("Before Training the Model")
    for inputs_k in A:
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
        flattened_A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        A.shape[0],
        A.shape[1],
        gamma
    )

    print("After Training the Model")
    for inputs_k in A:
        print(my_lib.RBF_predict_model_Classification(
            model,
            KM,
            numberOfCluser,
            gamma,
            inputs_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            len(inputs_k)
        ))


    my_lib.disposeRBF(model,KM)
    """






#CAS DE TEST 2 : Linear multiple
"""

    X = np.concatenate(
        [np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
    Y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])

    flattened_Y = Y.flatten()

    flattened_X = X.flatten()

    numberOfCluser = 2
    gamma = 0.001

    KM = my_lib.KMeans(numberOfCluser, flattened_X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), X.shape[0],
                       X.shape[1], 0, 3)


    model = my_lib.create_RBF_model(ctypes.c_int(numberOfCluser))

    for i in range(numberOfCluser*2):
        print("next")
        print(KM[i])


    print("Before Training the Model")
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
        flattened_Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
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
    
"""





#CAS DE TEST 3 : XOR
"""
X = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
Y = np.array([1, 1, -1, -1])
flattened_X = X.flatten()

numberOfCluser = 2
gamma = 0.01

KM = my_lib.KMeans(numberOfCluser, flattened_X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), X.shape[0],
                   X.shape[1], 0, 1)

model = my_lib.create_RBF_model(ctypes.c_int(numberOfCluser))

for i in range(numberOfCluser * 2):
    print("next")
    print(KM[i])

print("Before Training the Model")
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

my_lib.disposeRBF(model, KM)
"""




#CAS DE TEST 4 : CROSS (not work)
"""
X = np.random.random((500, 2)) * 2.0 - 1.0
Y = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in X])
flattened_X = X.flatten()

numberOfCluser = 2
gamma = 0.01

KM = my_lib.KMeans(numberOfCluser, flattened_X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), X.shape[0],
                   X.shape[1], -9, 9)

model = my_lib.create_RBF_model(ctypes.c_int(numberOfCluser))


for i in range(numberOfCluser * 2):
    print("next")
    print(KM[i])


print("Before Training the Model")
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


my_lib.disposeRBF(model, KM)
"""







#CAS DE TEST 1 : Linear Simple 2D (not work)
"""
X = np.array([
      [1],
      [2]
])
Y = np.array([
      2,
      3
])

flattened_X = X.flatten()

numberOfCluser = 2
gamma = 0.01

KM = my_lib.KMeans(numberOfCluser, flattened_X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), X.shape[0],
                   X.shape[1], 0, 2)

model = my_lib.create_RBF_model(ctypes.c_int(numberOfCluser))


for i in range(numberOfCluser * 2):
    print("next")
    print(KM[i])


print("Before Training the Model")
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


my_lib.disposeRBF(model, KM)
"""






#CAS DE TEST 2 : Non linear simple 2D (not work)
"""
X = np.array([
      [1],
      [2],
      [3]
])
Y = np.array([
      2,
      3,
      2.5
])

flattened_X = X.flatten()

numberOfCluser = 2
gamma = 0.01

KM = my_lib.KMeans(numberOfCluser, flattened_X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), X.shape[0],
                   X.shape[1], 1, 3)

model = my_lib.create_RBF_model(ctypes.c_int(numberOfCluser))


for i in range(numberOfCluser * 2):
    print("next")
    print(KM[i])


print("Before Training the Model")
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


my_lib.disposeRBF(model, KM)
"""





#CAS DE TEST 3: Linear simple 3D (not work)
"""
X = np.array([
      [1, 1],
      [2, 2],
      [3, 1]
])
Y = np.array([
      2,
      3,
      2.5
])

flattened_X = X.flatten()

numberOfCluser = 2
gamma = 0.1

KM = my_lib.KMeans(numberOfCluser, flattened_X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), X.shape[0],
                   X.shape[1], 1, 3)

model = my_lib.create_RBF_model(ctypes.c_int(numberOfCluser))


for i in range(numberOfCluser * 2):
    print("next")
    print(KM[i])


print("Before Training the Model")
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


my_lib.disposeRBF(model, KM)
"""