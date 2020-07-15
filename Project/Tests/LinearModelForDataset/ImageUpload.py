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
im16 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/16.jpeg")
im17 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/17.jpeg")
im18 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/18.jpeg")
im19 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/19.jpeg")
im20 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/20.jpeg")
im21 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/21.jpeg")
im22 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/22.jpeg")
im23 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/23.jpeg")
im24 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/24.jpeg")
im25 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/25.jpeg")
im26 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/26.jpeg")
im27 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/27.jpeg")
im28 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/28.jpeg")
im29 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/29.jpeg")
im30 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/30.jpeg")
im31 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/31.jpeg")
im32 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/32.jpeg")
im33 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/33.jpeg")
im34 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/34.jpeg")
im35 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/35.jpeg")
im36 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/36.jpeg")
im37 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/37.jpeg")
im38 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/38.jpeg")
im39 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/39.jpeg")
im40 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/40.jpeg")
im41 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/41.jpeg")
im42 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/42.jpeg")
im43 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/43.jpeg")
im44 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/44.jpeg")
im45 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/45.jpeg")
im46 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/46.jpeg")
im47 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/47.jpeg")
im48 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/48.jpeg")
im49 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/49.jpeg")
im50 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/50.jpeg")
im51 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/51.jpeg")
im52 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/52.jpeg")
im53 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/53.jpeg")
im54 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/54.jpeg")
im55 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/55.jpeg")
im56 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/56.jpeg")
im57 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/57.jpeg")
im58 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/58.jpeg")
im59 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/59.jpeg")
im60 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/60.jpeg")
im61 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/61.jpeg")
im62 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/62.jpeg")
im63 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/63.jpeg")
im64 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/64.jpeg")
im65 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/65.jpeg")
im66 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/66.jpeg")
im67 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/67.jpeg")
im68 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/68.jpeg")
im69 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/69.jpeg")
im70 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/70.jpeg")
im71 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/71.jpeg")
im72 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/72.jpeg")
im73 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/73.jpeg")
im74 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/74.jpeg")
im75 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/75.jpeg")
im76 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/76.jpeg")
im77 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/77.jpeg")
im78 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/78.jpeg")
im79 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/79.jpeg")
im80 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/80.jpeg")
im81 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/81.jpeg")
im82 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/82.jpeg")
im83 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/83.jpeg")
im84 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/84.jpeg")
im85 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/85.jpeg")
im86 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/86.jpeg")
im87 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/87.jpeg")
im88 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/88.jpeg")
im89 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/89.jpeg")
im90 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/90.jpeg")
im91 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/91.jpeg")
im92 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/92.jpeg")
im93 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/93.jpeg")
im94 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/94.jpeg")
im95 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/95.jpeg")
im96 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/96.jpeg")
im97 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/97.jpeg")
im98 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/98.jpeg")
im99 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/99.jpeg")
im100 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/100.jpeg")
im101 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/101.jpg")
im102 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/102.jpg")
im103 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/103.jpg")
im104 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/104.jpg")
im105 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/105.jpg")
im106 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/106.jpg")
im107 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/107.jpg")
im108 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/108.jpg")
im109 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/109.jpg")
im110 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/110.jpg")
im111 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/111.jpg")
im112 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/112.jpg")
im113 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/113.jpg")
im114 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/114.jpg")
im115 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/115.jpg")
im116 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/116.jpg")
im117 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/117.jpg")
im118 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/118.jpg")
im119 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/119.jpg")
im120 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/120.jpg")
im121 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/121.jpg")
im122 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/122.jpg")
im123 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/123.jpg")
im124 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/124.jpg")
im125 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/125.jpg")
im126 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/126.jpg")
im127 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/127.jpg")
im128 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/128.jpg")
im129 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/129.jpg")
im130 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/130.jpg")
im131 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/131.jpg")
im132 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/132.jpg")
im133 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/133.jpg")
im134 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/134.jpg")
im135 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/135.jpg")
im136 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/136.jpg")
im137 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/137.jpg")
im138 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/138.jpg")
im139 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/139.jpg")
im140 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/140.jpg")
im141 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/141.jpg")
im142 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/142.jpg")
im143 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/143.jpg")
im144 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/144.jpg")
im145 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/145.jpg")
im146 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/146.jpg")
im147 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/147.jpg")
im148 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/148.jpg")
im149 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/149.jpg")
im150 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/150.jpg")
im151 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/151.jpg")
im152 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/152.jpg")
im153 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/153.jpg")
#not happy
imN1 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/1.jpg")
imN2 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/2.jpg")
imN3 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/1.jpg")
imN4 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/2.jpg")
imN5 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/103.jpg")
imN6 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/104.jpg")
imN7 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/103.jpg")
imN8 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/104.jpg")
imN9 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/101.jpg")
imN10 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/102.jpg")
imN11 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/101.jpg")
imN12 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/102.jpg")


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
im_arr28 = np.array(im28) / 255.0
im_arr29 = np.array(im29) / 255.0
im_arr30 = np.array(im30) / 255.0
im_arr31 = np.array(im31) / 255.0
im_arr32 = np.array(im32) / 255.0
im_arr33 = np.array(im33) / 255.0
im_arr34 = np.array(im34) / 255.0
im_arr35 = np.array(im35) / 255.0
im_arr36 = np.array(im36) / 255.0
im_arr37 = np.array(im37) / 255.0
im_arr38 = np.array(im38) / 255.0
im_arr39 = np.array(im39) / 255.0
im_arr40 = np.array(im40) / 255.0
im_arr41 = np.array(im41) / 255.0
im_arr42 = np.array(im42) / 255.0
im_arr43 = np.array(im43) / 255.0
im_arr44 = np.array(im44) / 255.0
im_arr45 = np.array(im45) / 255.0
im_arr46 = np.array(im46) / 255.0
im_arr47 = np.array(im47) / 255.0
im_arr48 = np.array(im48) / 255.0
im_arr49 = np.array(im49) / 255.0
im_arr50 = np.array(im50) / 255.0
im_arr51 = np.array(im51) / 255.0
im_arr52 = np.array(im52) / 255.0
im_arr53 = np.array(im53) / 255.0
im_arr54 = np.array(im54) / 255.0
im_arr55 = np.array(im55) / 255.0
im_arr56 = np.array(im56) / 255.0
im_arr57 = np.array(im57) / 255.0
im_arr58 = np.array(im58) / 255.0
im_arr59 = np.array(im59) / 255.0
im_arr60 = np.array(im60) / 255.0
im_arr61 = np.array(im61) / 255.0
im_arr62 = np.array(im62) / 255.0
im_arr63 = np.array(im63) / 255.0
im_arr64 = np.array(im64) / 255.0
im_arr65 = np.array(im65) / 255.0
im_arr66 = np.array(im66) / 255.0
im_arr67 = np.array(im67) / 255.0
im_arr68 = np.array(im68) / 255.0
im_arr69 = np.array(im69) / 255.0
im_arr70 = np.array(im70) / 255.0
#not happy
im_arrN1 = np.array(imN1) / 255.0
im_arrN2 = np.array(imN2) / 255.0
im_arrN3 = np.array(imN3) / 255.0
im_arrN4 = np.array(imN4) / 255.0
im_arrN5 = np.array(imN5) / 255.0
im_arrN6 = np.array(imN6) / 255.0
im_arrN7 = np.array(imN7) / 255.0
im_arrN8 = np.array(imN8) / 255.0
im_arrN9 = np.array(imN9) / 255.0
im_arrN10 = np.array(imN10) / 255.0
im_arrN11 = np.array(imN11) / 255.0
im_arrN12 = np.array(imN12) / 255.0

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
image28 = im_arr28.flatten()
image29 = im_arr29.flatten()
image30 = im_arr30.flatten()
image31 = im_arr31.flatten()
image32 = im_arr32.flatten()
image33 = im_arr33.flatten()
image34 = im_arr34.flatten()
image35 = im_arr35.flatten()
image36 = im_arr36.flatten()
image37 = im_arr37.flatten()
image38 = im_arr38.flatten()
image39 = im_arr39.flatten()
image40 = im_arr40.flatten()
#not happy
imageN1 = im_arrN1.flatten()
imageN2 = im_arrN2.flatten()
imageN3 = im_arrN3.flatten()
imageN4 = im_arrN4.flatten()
imageN5 = im_arrN5.flatten()
imageN6 = im_arrN6.flatten()
imageN7 = im_arrN7.flatten()
imageN8 = im_arrN8.flatten()
imageN9 = im_arrN9.flatten()
imageN10 = im_arrN10.flatten()
imageN11 = im_arrN11.flatten()
imageN12 = im_arrN12.flatten()


dataset = np.array([image1, image2, image3, image4, image5, image6, image7, image8, image9, image10, image11,
                    image12, image13, image14, image15, image16, image17, image18, image19, image20, image21,
                    image22, image23, image24, image25, image26, image27, image28, image29, image30, image31,
                    image32, image33, image34, image35, image36, image37, image38, image39, image40, imageN1,
                    imageN2, imageN3, imageN4, imageN5, imageN6,
                    imageN7, imageN8, imageN9, imageN10, imageN11, imageN12], dtype='float64')

print(dataset.shape)


dataset_expected_output = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
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


