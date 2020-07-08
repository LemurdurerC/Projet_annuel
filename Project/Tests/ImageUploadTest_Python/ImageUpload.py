from PIL import Image
import numpy as np
import ctypes

import matplotlib.pyplot as plt

if __name__ == "__main__":
    path_to_dll = "../../Lib/LinearModelCppLib/cmake-build-debug" \
                  "/LinearModelCppLib.dll "
    """
    path_to_dll = "C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Lib/LinearModelCppLib/cmake-build-debug" \
                  "/LinearModelCppLib.dll "
 """
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

im = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/1.jpg")
im2 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/2.jpg")
im3 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/3.jpg")
im4 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/4.jpg")
im9 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/1.jpg")
im10 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/2.jpg")
im11 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/1.jpg")
im12 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/2.jpg")

im_arr1 = np.array(im) / 255.0
im_arr2 = np.array(im2) / 255.0
im_arr3 = np.array(im3) / 255.0
im_arr4 = np.array(im4) / 255.0
im_arr9 = np.array(im9) / 255.0
im_arr10 = np.array(im10) / 255.0
im_arr11 = np.array(im11) / 255.0
im_arr12 = np.array(im12) / 255.0

image1 = im_arr1.flatten()
image2 = im_arr2.flatten()
image3 = im_arr3.flatten()
image4 = im_arr4.flatten()
image9 = im_arr9.flatten()
image10 = im_arr10.flatten()
image11 = im_arr11.flatten()
image12 = im_arr12.flatten()


dataset = np.array([image1, image12, image3, image4, image9, image10, image11, image2], dtype='float64')
print(dataset.shape)

dataset_expected_output = np.array([1, -1, 1, 1, -1, -1, -1, 1], dtype='float64')


model = my_lib.linear_create_model(ctypes.c_int(dataset.shape[1]))


flattened_Dataset = dataset.flatten()

print("Before Training")
for inputs_k in dataset:
    print(my_lib.linear_predict_model_classification(
        model,
        inputs_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(inputs_k)

    ))


my_lib.linear_train_model_classification(
    model,
    flattened_Dataset.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    dataset_expected_output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    dataset.shape[0],
    dataset.shape[1],
    0.001,
    10000
)

print("After Training......")

for inputs_k in dataset:
    print(my_lib.linear_predict_model_classification(
        model,
        inputs_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(inputs_k)
    ))

my_lib.linear_dispose_model(model)

"""
rouge_1 = im_arr1[:,:,0]
#vert_1 = im_arr1[:,:,1]
#bleu_1 = im_arr1[:,:,2]



rouge_2 = im_arr2[:,:,0]
#vert_2 = im_arr2[:,:,1]
#bleu_2 = im_arr2[:,:,2]


rouge_3 = im_arr3[:, :, 0]
rouge_4 = im_arr4[:, :, 0]
rouge_9 = im_arr9[:, :, 0]
rouge_10 = im_arr10[:, :, 0]
rouge_11 = im_arr11[:, :, 0]
rouge_12 = im_arr12[:, :, 0]




r1 = rouge_1.flatten()
r2 = rouge_2.flatten()
r3 = rouge_3.flatten()
r4 = rouge_4.flatten()
r9 = rouge_9.flatten()
r10 = rouge_10.flatten()
r11 = rouge_11.flatten()
r12 = rouge_12.flatten()


im_arr = np.array([r1, r2, r3, r4, r9, r10, r11, r12])

im_arr_expec = np.array([1, 1, 1, 1, -1, -1, -1, -1])

A = im_arr.flatten()


model = my_lib.linear_create_model(ctypes.c_int(3300))


for inputs_k in im_arr:
    print(my_lib.linear_predict_model_classification(
                model,
                inputs_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                len(inputs_k)
            ))



my_lib.linear_train_model_classification(
        model,
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        im_arr_expec.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        im_arr.shape[0],
        im_arr.shape[1],
        0.01,
        1000
)

print("AFTER")

for inputs_k in im_arr:
    print(my_lib.linear_predict_model_classification(
                model,
                inputs_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                len(inputs_k)
            ))



my_lib.linear_dispose_model(model)



"""
