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
im5 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/5.jpg")
im6 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/6.jpg")
im7 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/7.jpg")
im8 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/8.jpg")
im9 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/9.jpg")
im10 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/10.jpg")
im11 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/11.jpeg")
im12 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/12.jpeg")
im13 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/13.jpeg")
im14 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/14.jpeg")
im15 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/15.jpeg")
#not happy
im16 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/1.jpg")
im17 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/2.jpg")
im18 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/1.jpg")
im19 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/2.jpg")
im20 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/103.jpg")
im21 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/104.jpg")
im22 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/103.jpg")
im23 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/104.jpg")
im24 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/101.jpg")
im25 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/102.jpg")
im26 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/101.jpg")
im27 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/102.jpg")


im_arr1 = np.array(im) / 255.0
im_arr2 = np.array(im2) / 255.0
im_arr3 = np.array(im3) / 255.0
im_arr4 = np.array(im4) / 255.0
im_arr5 = np.array(im5) / 255.0
im_arr6 = np.array(im6) / 255.0
im_arr7 = np.array(im7) / 255.0
im_arr8 = np.array(im8) / 255.0
im_arr9 = np.array(im9) / 255.0
im_arr10 = np.array(im10) / 255.0
im_arr11 = np.array(im11) / 255.0
im_arr12 = np.array(im12) / 255.0
im_arr13 = np.array(im13) / 255.0
im_arr14 = np.array(im14) / 255.0
im_arr15 = np.array(im15) / 255.0
im_arr16 = np.array(im16) / 255.0
im_arr17 = np.array(im17) / 255.0
im_arr18 = np.array(im18) / 255.0
im_arr19 = np.array(im19) / 255.0
im_arr20 = np.array(im20) / 255.0
im_arr21 = np.array(im21) / 255.0
im_arr22 = np.array(im22) / 255.0
im_arr23 = np.array(im23) / 255.0
im_arr24 = np.array(im24) / 255.0
im_arr25 = np.array(im25) / 255.0
im_arr26 = np.array(im26) / 255.0
im_arr27 = np.array(im27) / 255.0

image1 = im_arr1.flatten()
image2 = im_arr2.flatten()
image3 = im_arr3.flatten()
image4 = im_arr4.flatten()
image5 = im_arr5.flatten()
image6 = im_arr6.flatten()
image7 = im_arr7.flatten()
image8 = im_arr8.flatten()
image9 = im_arr9.flatten()
image10 = im_arr10.flatten()
image11 = im_arr11.flatten()
image12 = im_arr12.flatten()
image13 = im_arr13.flatten()
image14 = im_arr14.flatten()
image15 = im_arr15.flatten()
image16 = im_arr16.flatten()
image17 = im_arr17.flatten()
image18 = im_arr18.flatten()
image19 = im_arr19.flatten()
image20 = im_arr20.flatten()
image21 = im_arr21.flatten()
image22 = im_arr22.flatten()
image23 = im_arr23.flatten()
image24 = im_arr24.flatten()
image25 = im_arr25.flatten()
image26 = im_arr26.flatten()
image27 = im_arr27.flatten()


dataset = np.array([image1, image2, image3, image4, image5, image6, image7, image8, image9, image10, image11,
                    image12, image13, image14, image15, image16, image17, image18, image19, image20, image21,
                    image22, image23, image24, image25, image26, image27], dtype='float64')

print(dataset.shape)


dataset_expected_output = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], dtype='float64')


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




print("Et des photos non connnus ?")

imTest = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Test/Happy/1.jpeg")
imTest2 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Test/Happy/10.jpeg")
im_arrTest = np.array(imTest) / 255.0
imageTest = im_arrTest.flatten()
im_arrTest2 = np.array(imTest2) / 255.0
imageTest2 = im_arrTest2.flatten()
print(my_lib.linear_predict_model_classification(
        model,
        imageTest.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(imageTest)
    ))
print(my_lib.linear_predict_model_classification(
        model,
        imageTest2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        len(imageTest2)
    ))


