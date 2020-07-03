import ctypes
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


if __name__ == "__main__":

    path_to_dll = "../../Lib/RBFLib/cmake-build-debug/RBFLib.dll"

    my_lib = ctypes.CDLL(path_to_dll)

    my_lib.create_RBF_model.argtypes = [ctypes.c_int]
    my_lib.create_RBF_model.restype = ctypes.c_void_p

    my_lib.disposeRBF.argtypes = [ctypes.c_void_p,ctypes.c_void_p]
    my_lib.disposeRBF.restype = None

    my_lib.RBF_predict_model_Regression.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int
    ]
    my_lib.RBF_predict_model_Regression.restype = ctypes.c_double

    my_lib.RBF_predict_model_Classification.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
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
        ctypes.c_int
    ]
    my_lib.RBF_train_model.restype = None


    print("Hello");