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
im51 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/51.jpg")
im52 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/52.jpg")
im53 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/53.jpg")
im54 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/54.jpg")
im55 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/55.jpg")
im56 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/56.jpg")
im57 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/57.jpg")
im58 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/58.jpg")
im59 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/59.jpg")
im60 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/60.jpg")
im61 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/61.jpg")
im62 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/62.jpg")
im63 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/63.jpg")
im64 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/64.jpg")
im65 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/65.jpg")
im66 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/66.jpg")
im67 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/67.jpg")
im68 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/68.jpg")
im69 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/69.jpg")
im70 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/70.jpg")
im71 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/71.jpg")
im72 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/72.jpg")
im73 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/73.jpg")
im74 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/74.jpg")
im75 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/75.jpg")
im76 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/76.jpg")
im77 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/77.jpg")
im78 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/78.jpg")
im79 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/79.jpg")
im80 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/80.jpg")
im81 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/81.jpg")
im82 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/82.jpg")
im83 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/83.jpg")
im84 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/84.jpg")
im85 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/85.jpg")
im86 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/86.jpg")
im87 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/87.jpg")
im88 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/88.jpg")
im89 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/89.jpg")
im90 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/90.jpg")
im91 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/91.jpg")
im92 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/92.jpg")
im93 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/93.jpg")
im94 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/94.jpg")
im95 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/95.jpg")
im96 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/96.jpg")
im97 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/97.jpg")
im98 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/98.jpg")
im99 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/99.jpg")
im100 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/100.jpg")
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
imN1 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/1.jpg")
imN2 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/2.jpg")
imN3 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/3.jpg")
imN4 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/4.jpg")
imN5 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/5.jpg")
imN6 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/6.jpg")
imN7 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/7.jpg")
imN8 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/8.jpg")
imN9 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/9.jpeg")
imN10 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/10.jpeg")
imN11 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/11.jpeg")
imN12 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/12.jpeg")
imN13 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/13.jpeg")
imN14 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/14.jpeg")
imN15 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/15.jpeg")
imN16 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/16.jpeg")
imN17 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/17.jpeg")
imN18 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/18.jpeg")
imN19 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/19.jpeg")
imN20 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/20.jpeg")
imN21 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/21.jpeg")
imN22 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/22.jpeg")
imN23 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/23.jpeg")
imN24 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/24.jpeg")
imN25 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/25.jpeg")
imN26 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/26.jpeg")
imN27 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/27.jpeg")
imN28 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/28.jpeg")
imN29 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/29.jpeg")
imN30 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/30.jpeg")
imN31 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/31.jpeg")
imN32 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/32.jpeg")
imN33 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/33.jpeg")
imN34 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/34.jpeg")
imN35 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/35.jpeg")
imN36 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/36.jpeg")
imN37 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/37.jpeg")
imN38 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/38.jpeg")
imN39 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/39.jpeg")
imN40 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/40.jpeg")
imN41 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/41.jpeg")
imN42 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/42.jpeg")
imN43 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/43.jpeg")
imN44 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/44.jpeg")
imN45 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/45.jpeg")
imN46 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/46.jpeg")
imN47 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/47.jpeg")
imN48 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/48.jpeg")
imN49 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/49.jpeg")
imN50 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/50.jpeg")
imN51 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/51.jpg")
imN52 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/52.jpg")
imN53 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/53.jpg")
imN54 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/54.jpg")
imN55 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/55.jpg")
imN56 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/56.jpg")
imN57 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/57.jpg")
imN58 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/58.jpg")
imN59 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/59.jpg")
imN60 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/60.jpg")
imN61 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/61.jpg")
imN62 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/62.jpg")
imN63 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/63.jpg")
imN64 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/64.jpg")
imN65 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/65.jpg")
imN66 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/66.jpg")
imN67 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/67.jpg")
imN68 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/68.jpg")
imN69 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/69.jpg")
imN70 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/70.jpg")
imN71 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/71.jpg")
imN72 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/72.jpg")
imN73 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/73.jpg")
imN74 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/74.jpg")
imN75 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/75.jpg")
imN76 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/76.jpg")
imN77 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/77.jpg")
imN78 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/78.jpg")
imN79 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/79.jpg")
imN80 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/80.jpg")
imN81 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/81.jpg")
imN82 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/82.jpg")
imN83 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/83.jpg")
imN84 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/84.jpg")
imN85 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/85.jpg")
imN86 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/86.jpg")
imN87 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/87.jpg")
imN88 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/88.jpg")
imN89 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/89.jpg")
imN90 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/90.jpg")
imN91 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/91.jpg")
imN92 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/92.jpg")
imN93 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/93.jpg")
imN94 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/94.jpg")
imN95 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/95.jpg")
imN96 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/96.jpg")
imN97 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/97.jpg")
imN98 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/98.jpg")
imN99 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/99.jpg")
imN100 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/100.jpg")
imN101 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/101.jpg")
imN102 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/102.jpg")
imN103 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/103.jpg")
imN104 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/104.jpg")
imN105 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/105.jpg")
imN106 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/106.jpg")
imN107 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/107.jpg")
imN108 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/108.jpg")
imN109 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/109.jpg")
imN110 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/110.jpg")
imN111 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/111.jpg")
imN112 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/112.jpg")
imN113 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/113.jpg")
imN114 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/114.jpg")
imN115 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/115.jpg")
imN116 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/116.jpg")
imN117 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/117.jpg")
imN118 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/118.jpg")
imN119 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/119.jpg")
imN120 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/120.jpg")
imN121 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/121.jpg")
imN122 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/122.jpg")
imN123 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/123.jpg")
imN124 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/124.jpg")
imN125 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/125.jpg")
imN126 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/126.jpg")
imN127 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/127.jpg")
imN128 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/128.jpg")
imN129 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/129.jpg")
imN130 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/130.jpg")
imN131 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/131.jpg")
imN132 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/132.jpg")
imN133 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/133.jpg")
imN134 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/134.jpg")
imN135 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/135.jpg")
imN136 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/136.jpg")
imN137 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/137.jpg")
imN138 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/138.jpg")
imN139 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/139.jpg")
imN140 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/140.jpg")
imN141 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/141.jpg")
imN142 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/142.jpg")
imN143 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/143.jpg")
imN144 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/144.jpg")
imN145 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/145.jpg")
imN146 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/146.jpg")
imN147 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/147.jpg")
imN148 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/148.jpg")
imN149 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/149.jpg")
imN150 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/150.jpg")
imN151 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/151.jpg")
imN152 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/152.jpg")


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
im_arr71 = np.array(im71) / 255.0
im_arr72 = np.array(im72) / 255.0
im_arr73 = np.array(im73) / 255.0
im_arr74 = np.array(im74) / 255.0
im_arr75 = np.array(im75) / 255.0
im_arr76 = np.array(im76) / 255.0
im_arr77 = np.array(im77) / 255.0
im_arr78 = np.array(im78) / 255.0
im_arr79 = np.array(im79) / 255.0
im_arr80 = np.array(im80) / 255.0
im_arr81 = np.array(im81) / 255.0
im_arr82 = np.array(im82) / 255.0
im_arr83 = np.array(im83) / 255.0
im_arr84 = np.array(im84) / 255.0
im_arr85 = np.array(im85) / 255.0
im_arr86 = np.array(im86) / 255.0
im_arr87 = np.array(im87) / 255.0
im_arr88 = np.array(im88) / 255.0
im_arr89 = np.array(im89) / 255.0
im_arr90 = np.array(im90) / 255.0
im_arr91 = np.array(im91) / 255.0
im_arr92 = np.array(im92) / 255.0
im_arr93 = np.array(im93) / 255.0
im_arr94 = np.array(im94) / 255.0
im_arr95 = np.array(im95) / 255.0
im_arr96 = np.array(im96) / 255.0
im_arr97 = np.array(im97) / 255.0
im_arr98 = np.array(im98) / 255.0
im_arr99 = np.array(im99) / 255.0
im_arr100 = np.array(im100) / 255.0
im_arr101 = np.array(im101) / 255.0
im_arr102 = np.array(im102) / 255.0
im_arr103 = np.array(im103) / 255.0
im_arr104 = np.array(im104) / 255.0
im_arr105 = np.array(im105) / 255.0
im_arr106 = np.array(im106) / 255.0
im_arr107 = np.array(im107) / 255.0
im_arr108 = np.array(im108) / 255.0
im_arr109 = np.array(im109) / 255.0
im_arr110 = np.array(im110) / 255.0
im_arr111 = np.array(im111) / 255.0
im_arr112 = np.array(im112) / 255.0
im_arr113 = np.array(im113) / 255.0
im_arr114 = np.array(im114) / 255.0
im_arr115 = np.array(im115) / 255.0
im_arr116 = np.array(im116) / 255.0
im_arr117 = np.array(im117) / 255.0
im_arr118 = np.array(im118) / 255.0
im_arr119 = np.array(im119) / 255.0
im_arr120 = np.array(im120) / 255.0
im_arr121 = np.array(im121) / 255.0
im_arr122 = np.array(im122) / 255.0
im_arr123 = np.array(im123) / 255.0
im_arr124 = np.array(im124) / 255.0
im_arr125 = np.array(im125) / 255.0
im_arr126 = np.array(im126) / 255.0
im_arr127 = np.array(im127) / 255.0
im_arr128 = np.array(im128) / 255.0
im_arr129 = np.array(im129) / 255.0
im_arr130 = np.array(im130) / 255.0
im_arr131 = np.array(im131) / 255.0
im_arr132 = np.array(im132) / 255.0
im_arr133 = np.array(im133) / 255.0
im_arr134 = np.array(im134) / 255.0
im_arr135 = np.array(im135) / 255.0
im_arr136 = np.array(im136) / 255.0
im_arr137 = np.array(im137) / 255.0
im_arr138 = np.array(im138) / 255.0
im_arr139 = np.array(im139) / 255.0
im_arr140 = np.array(im140) / 255.0
im_arr141 = np.array(im141) / 255.0
im_arr142 = np.array(im142) / 255.0
im_arr143 = np.array(im143) / 255.0
im_arr144 = np.array(im144) / 255.0
im_arr145 = np.array(im145) / 255.0
im_arr146 = np.array(im146) / 255.0
im_arr147 = np.array(im147) / 255.0
im_arr148 = np.array(im148) / 255.0
im_arr149 = np.array(im149) / 255.0
im_arr150 = np.array(im150) / 255.0
im_arr151 = np.array(im151) / 255.0
im_arr152 = np.array(im152) / 255.0
im_arr153 = np.array(im153) / 255.0
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
image41 = im_arr41.flatten()
image42 = im_arr42.flatten()
image43 = im_arr43.flatten()
image44 = im_arr44.flatten()
image45 = im_arr45.flatten()
image46 = im_arr46.flatten()
image47 = im_arr47.flatten()
image48 = im_arr48.flatten()
image49 = im_arr49.flatten()
image50 = im_arr50.flatten()
image51 = im_arr51.flatten()
image52 = im_arr52.flatten()
image53 = im_arr53.flatten()
image54 = im_arr54.flatten()
image55 = im_arr55.flatten()
image56 = im_arr56.flatten()
image57 = im_arr57.flatten()
image58 = im_arr58.flatten()
image59 = im_arr59.flatten()
image60 = im_arr60.flatten()
image61 = im_arr61.flatten()
image62 = im_arr62.flatten()
image63 = im_arr63.flatten()
image64 = im_arr64.flatten()
image65 = im_arr65.flatten()
image66 = im_arr66.flatten()
image67 = im_arr67.flatten()
image68 = im_arr68.flatten()
image69 = im_arr69.flatten()
image70 = im_arr70.flatten()
image71 = im_arr71.flatten()
image72 = im_arr72.flatten()
image73 = im_arr73.flatten()
image74 = im_arr74.flatten()
image75 = im_arr75.flatten()
image76 = im_arr76.flatten()
image77 = im_arr77.flatten()
image78 = im_arr78.flatten()
image79 = im_arr79.flatten()
image80 = im_arr80.flatten()
image81 = im_arr81.flatten()
image82 = im_arr82.flatten()
image83 = im_arr83.flatten()
image84 = im_arr84.flatten()
image85 = im_arr85.flatten()
image86 = im_arr86.flatten()
image87 = im_arr87.flatten()
image88 = im_arr88.flatten()
image89 = im_arr89.flatten()
image90 = im_arr90.flatten()
image91 = im_arr91.flatten()
image92 = im_arr92.flatten()
image93 = im_arr93.flatten()
image94 = im_arr94.flatten()
image95 = im_arr95.flatten()
image96 = im_arr96.flatten()
image97 = im_arr97.flatten()
image98 = im_arr98.flatten()
image99 = im_arr99.flatten()
image100 = im_arr100.flatten()
image101 = im_arr101.flatten()
image102 = im_arr102.flatten()
image103 = im_arr103.flatten()
image104 = im_arr104.flatten()
image105 = im_arr105.flatten()
image106 = im_arr106.flatten()
image107 = im_arr107.flatten()
image108 = im_arr108.flatten()
image109 = im_arr109.flatten()
image110 = im_arr110.flatten()
image111 = im_arr111.flatten()
image112 = im_arr112.flatten()
image113 = im_arr113.flatten()
image114 = im_arr114.flatten()
image115 = im_arr115.flatten()
image116 = im_arr116.flatten()
image117 = im_arr117.flatten()
image118 = im_arr118.flatten()
image119 = im_arr119.flatten()
image120 = im_arr120.flatten()
image121 = im_arr121.flatten()
image122 = im_arr122.flatten()
image123 = im_arr123.flatten()
image124 = im_arr124.flatten()
image125 = im_arr125.flatten()
image126 = im_arr126.flatten()
image127 = im_arr127.flatten()
image128 = im_arr128.flatten()
image129 = im_arr129.flatten()
image130 = im_arr130.flatten()
image131 = im_arr131.flatten()
image132 = im_arr132.flatten()
image133 = im_arr133.flatten()
image134 = im_arr134.flatten()
image135 = im_arr135.flatten()
image136 = im_arr136.flatten()
image137 = im_arr137.flatten()
image138 = im_arr138.flatten()
image139 = im_arr139.flatten()
image140 = im_arr140.flatten()
image141 = im_arr141.flatten()
image142 = im_arr142.flatten()
image143 = im_arr143.flatten()
image144 = im_arr144.flatten()
image145 = im_arr145.flatten()
image146 = im_arr146.flatten()
image147 = im_arr147.flatten()
image148 = im_arr148.flatten()
image149 = im_arr149.flatten()
image150 = im_arr150.flatten()
image151 = im_arr151.flatten()
image152 = im_arr152.flatten()
image153 = im_arr153.flatten()
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


