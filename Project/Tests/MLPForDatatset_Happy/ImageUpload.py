from PIL import Image
import ctypes
import numpy as np


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

#region
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
im154 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/154.jpg")
im155 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/155.jpg")
im156 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/156.jpg")
im157 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/157.jpg")
im158 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/158.jpg")
im159 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/159.jpg")
im160 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/160.jpg")
im161 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/161.jpg")
im162 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/162.jpg")
im163 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/163.jpg")
im164 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/164.jpg")
im165 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/165.jpg")
im166 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/166.jpg")
im167 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/167.jpg")
im168 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/168.jpg")
im169 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/169.jpg")
im170 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/170.jpg")
im171 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/171.jpg")
im172 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/172.jpg")
im173 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/173.jpg")
im174 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/174.jpg")
im175 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/175.jpg")
im176 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/176.jpg")
im177 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/177.jpg")
im178 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/178.jpg")
im179 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/179.jpg")
im180 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/180.jpg")
im181 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/181.jpg")
im182 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/182.jpg")
im183 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/183.jpg")
im184 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/184.jpg")
im185 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/185.jpg")
im186 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/186.jpg")
im187 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/187.jpg")
im188 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/188.jpg")
im189 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/189.jpg")
im190 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/190.jpg")
im191 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/191.jpg")
im192 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/192.jpg")
im193 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/193.jpg")
im194 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/194.jpg")
im195 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/195.jpg")
im196 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/196.jpg")
im197 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/197.jpg")
im198 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/198.jpg")
im199 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/199.jpg")
im200 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Happy/200.jpg")
#NOT HAPPY
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
imN153 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/153.jpg")
imN154 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/154.jpg")
imN155 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/155.jpg")
imN156 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/156.jpg")
imN157 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/157.jpg")
imN158 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/158.jpg")
imN159 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/159.jpg")
imN160 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/160.jpg")
imN161 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/161.jpg")
imN162 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/162.jpg")
imN163 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/163.jpg")
imN164 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/164.jpg")
imN165 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/165.jpg")
imN166 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/166.jpg")
imN167 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/167.jpg")
imN168 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/168.jpg")
imN169 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/169.jpg")
imN170 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/170.jpg")
imN171 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/171.jpg")
imN172 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/172.jpg")
imN173 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/173.jpg")
imN174 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/174.jpg")
imN175 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/175.jpg")
imN176 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/176.jpg")
imN177 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/177.jpg")
imN178 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/178.jpg")
imN179 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/179.jpg")
imN180 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/180.jpg")
imN181 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/181.jpg")
imN182 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/182.jpg")
imN183 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/183.jpg")
imN184 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/184.jpg")
imN185 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/185.jpg")
imN186 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/186.jpg")
imN187 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/187.jpg")
imN188 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/188.jpg")
imN189 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/189.jpg")
imN190 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/190.jpg")
imN191 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/191.jpg")
imN192 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/192.jpg")
imN193 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/193.jpg")
imN194 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/194.jpg")
imN195 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/195.jpg")
imN196 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/196.jpg")
imN197 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/197.jpg")
imN198 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/198.jpg")
imN199 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/199.jpg")
imN200 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Neutral/200.jpg")
imA1 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/1.jpg")
imA2 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/2.jpg")
imA3 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/3.jpg")
imA4 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/4.jpg")
imA5 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/5.jpg")
imA6 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/6.jpg")
imA7 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/7.jpg")
imA8 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/8.jpg")
imA9 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/9.jpg")
imA10 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/10.jpeg")
imA11 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/11.jpeg")
imA12 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/12.jpeg")
imA13 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/13.jpeg")
imA14 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/14.jpeg")
imA15 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/15.jpeg")
imA16 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/16.jpeg")
imA17 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/17.jpeg")
imA18 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/18.jpeg")
imA19 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/19.jpeg")
imA20 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/20.jpeg")
imA21 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/21.jpeg")
imA22 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/22.jpeg")
imA23 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/23.jpeg")
imA24 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/24.jpeg")
imA25 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/25.jpeg")
imA26 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/26.jpeg")
imA27 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/27.jpeg")
imA28 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/28.jpeg")
imA29 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/29.jpeg")
imA30 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/30.jpeg")
imA31 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/31.jpeg")
imA32 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/32.jpeg")
imA33 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/33.jpeg")
imA34 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/34.jpeg")
imA35 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/35.jpeg")
imA36 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/36.jpeg")
imA37 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/37.jpeg")
imA38 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/38.jpeg")
imA39 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/39.jpeg")
imA40 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/40.jpeg")
imA41 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/41.jpeg")
imA42 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/42.jpeg")
imA43 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/43.jpeg")
imA44 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/44.jpeg")
imA45 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/45.jpeg")
imA46 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/46.jpeg")
imA47 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/47.jpeg")
imA48 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/48.jpeg")
imA49 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/49.jpeg")
imA50 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/50.jpeg")
imA51 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/51.jpg")
imA52 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/52.jpg")
imA53 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/53.jpg")
imA54 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/54.jpg")
imA55 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/55.jpg")
imA56 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/56.jpg")
imA57 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/57.jpg")
imA58 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/58.jpg")
imA59 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/59.jpg")
imA60 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/60.jpg")
imA61 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/61.jpg")
imA62 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/62.jpg")
imA63 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/63.jpg")
imA64 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/64.jpg")
imA65 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/65.jpg")
imA66 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/66.jpg")
imA67 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/67.jpg")
imA68 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/68.jpg")
imA69 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/69.jpg")
imA70 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/70.jpg")
imA71 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/71.jpg")
imA72 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/72.jpg")
imA73 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/73.jpg")
imA74 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/74.jpg")
imA75 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/75.jpg")
imA76 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/76.jpg")
imA77 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/77.jpg")
imA78 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/78.jpg")
imA79 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/79.jpg")
imA80 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/80.jpg")
imA81 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/81.jpg")
imA82 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/82.jpg")
imA83 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/83.jpg")
imA84 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/84.jpg")
imA85 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/85.jpg")
imA86 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/86.jpg")
imA87 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/87.jpg")
imA88 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/88.jpg")
imA89 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/89.jpg")
imA90 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/90.jpg")
imA91 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/91.jpg")
imA92 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/92.jpg")
imA93 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/93.jpg")
imA94 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/94.jpg")
imA95 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/95.jpg")
imA96 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/96.jpg")
imA97 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/97.jpg")
imA98 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/98.jpg")
imA99 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/99.jpg")
imA100 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/100.jpg")
imA101 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/101.jpg")
imA102 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/102.jpg")
imA103 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/103.jpg")
imA104 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/104.jpg")
imA105 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/105.jpg")
imA106 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/106.jpg")
imA107 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/107.jpg")
imA108 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/108.jpg")
imA109 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/109.jpg")
imA110 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/110.jpg")
imA111 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/111.jpg")
imA112 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/112.jpg")
imA113 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/113.jpg")
imA114 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/114.jpg")
imA115 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/115.jpg")
imA116 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/116.jpg")
imA117 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/117.jpg")
imA118 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/118.jpg")
imA119 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/119.jpg")
imA120 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/120.jpg")
imA121 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/121.jpg")
imA122 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/122.jpg")
imA123 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/123.jpg")
imA124 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/124.jpg")
imA125 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/125.jpg")
imA126 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/126.jpg")
imA127 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/127.jpg")
imA128 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/128.jpg")
imA129 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/129.jpg")
imA130 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/130.jpg")
imA131 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/131.jpg")
imA132 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/132.jpg")
imA133 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/133.jpg")
imA134 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/134.jpg")
imA135 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/135.jpg")
imA136 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/136.jpg")
imA137 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/137.jpg")
imA138 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/138.jpg")
imA139 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/139.jpg")
imA140 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/140.jpg")
imA141 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/141.jpg")
imA142 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/142.jpg")
imA143 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/143.jpg")
imA144 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/144.jpg")
imA145 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/145.jpg")
imA146 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/146.jpg")
imA147 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/147.jpg")
imA148 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/148.jpg")
imA149 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/149.jpg")
imA150 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/150.jpg")
imA151 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/151.jpg")
imA152 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/152.jpg")
imA153 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/153.jpg")
imA154 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/154.jpg")
imA155 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/155.jpg")
imA156 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/156.jpg")
imA157 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/157.jpg")
imA158 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/158.jpg")
imA159 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/159.jpg")
imA160 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/160.jpg")
imA161 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/161.jpg")
imA162 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/162.jpg")
imA163 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/163.jpg")
imA164 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/164.jpg")
imA165 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/165.jpg")
imA166 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/166.jpg")
imA167 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/167.jpg")
imA168 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/168.jpg")
imA169 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/169.jpg")
imA170 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/170.jpg")
imA171 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/171.jpg")
imA172 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/172.jpg")
imA173 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/173.jpg")
imA174 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/174.jpg")
imA175 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/175.jpg")
imA176 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/176.jpg")
imA177 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/177.jpg")
imA178 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/178.jpg")
imA179 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/179.jpg")
imA180 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/180.jpg")
imA181 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/181.jpg")
imA182 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/182.jpg")
imA183 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/183.jpg")
imA184 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/184.jpg")
imA185 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/185.jpg")
imA186 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/186.jpg")
imA187 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/187.jpg")
imA188 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/188.jpg")
imA189 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/189.jpg")
imA190 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/190.jpg")
imA191 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/191.jpg")
imA192 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/192.jpg")
imA193 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/193.jpg")
imA194 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/194.jpg")
imA195 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/195.jpg")
imA196 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/196.jpg")
imA197 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/197.jpg")
imA198 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/198.jpg")
imA199 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/199.jpg")
imA200 = Image.open("C:/Users/hejar/OneDrive/Bureau/PA_Git/Projet_annuel/Project/Dataset/Train/Angry/200.jpg")

#endregion

#region
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
im_arr154 = np.array(im154) / 255.0
im_arr155 = np.array(im155) / 255.0
im_arr156 = np.array(im156) / 255.0
im_arr157 = np.array(im157) / 255.0
im_arr158 = np.array(im158) / 255.0
im_arr159 = np.array(im159) / 255.0
im_arr160 = np.array(im160) / 255.0
im_arr161 = np.array(im161) / 255.0
im_arr162 = np.array(im162) / 255.0
im_arr163 = np.array(im163) / 255.0
im_arr164 = np.array(im164) / 255.0
im_arr165 = np.array(im165) / 255.0
im_arr166 = np.array(im166) / 255.0
im_arr167 = np.array(im167) / 255.0
im_arr168 = np.array(im168) / 255.0
im_arr169 = np.array(im169) / 255.0
im_arr170 = np.array(im170) / 255.0
im_arr171 = np.array(im171) / 255.0
im_arr172 = np.array(im172) / 255.0
im_arr173 = np.array(im173) / 255.0
im_arr174 = np.array(im174) / 255.0
im_arr175 = np.array(im175) / 255.0
im_arr176 = np.array(im176) / 255.0
im_arr177 = np.array(im177) / 255.0
im_arr178 = np.array(im178) / 255.0
im_arr179 = np.array(im179) / 255.0
im_arr180 = np.array(im180) / 255.0
im_arr181 = np.array(im181) / 255.0
im_arr182 = np.array(im182) / 255.0
im_arr183 = np.array(im183) / 255.0
im_arr184 = np.array(im184) / 255.0
im_arr185 = np.array(im185) / 255.0
im_arr186 = np.array(im186) / 255.0
im_arr187 = np.array(im187) / 255.0
im_arr188 = np.array(im188) / 255.0
im_arr189 = np.array(im189) / 255.0
im_arr190 = np.array(im190) / 255.0
im_arr191 = np.array(im191) / 255.0
im_arr192 = np.array(im192) / 255.0
im_arr193 = np.array(im193) / 255.0
im_arr194 = np.array(im194) / 255.0
im_arr195 = np.array(im195) / 255.0
im_arr196 = np.array(im196) / 255.0
im_arr197 = np.array(im197) / 255.0
im_arr198 = np.array(im198) / 255.0
im_arr199 = np.array(im199) / 255.0
im_arr200 = np.array(im200) / 255.0
#NOT HAPPY
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
im_arrN13 = np.array(imN13) / 255.0
im_arrN14 = np.array(imN14) / 255.0
im_arrN15 = np.array(imN15) / 255.0
im_arrN16 = np.array(imN16) / 255.0
im_arrN17 = np.array(imN17) / 255.0
im_arrN18 = np.array(imN18) / 255.0
im_arrN19 = np.array(imN19) / 255.0
im_arrN20 = np.array(imN20) / 255.0
im_arrN21 = np.array(imN21) / 255.0
im_arrN22 = np.array(imN22) / 255.0
im_arrN23 = np.array(imN23) / 255.0
im_arrN24 = np.array(imN24) / 255.0
im_arrN25 = np.array(imN25) / 255.0
im_arrN26 = np.array(imN26) / 255.0
im_arrN27 = np.array(imN27) / 255.0
im_arrN28 = np.array(imN28) / 255.0
im_arrN29 = np.array(imN29) / 255.0
im_arrN30 = np.array(imN30) / 255.0
im_arrN31 = np.array(imN31) / 255.0
im_arrN32 = np.array(imN32) / 255.0
im_arrN33 = np.array(imN33) / 255.0
im_arrN34 = np.array(imN34) / 255.0
im_arrN35 = np.array(imN35) / 255.0
im_arrN36 = np.array(imN36) / 255.0
im_arrN37 = np.array(imN37) / 255.0
im_arrN38 = np.array(imN38) / 255.0
im_arrN39 = np.array(imN39) / 255.0
im_arrN40 = np.array(imN40) / 255.0
im_arrN41 = np.array(imN41) / 255.0
im_arrN42 = np.array(imN42) / 255.0
im_arrN43 = np.array(imN43) / 255.0
im_arrN44 = np.array(imN44) / 255.0
im_arrN45 = np.array(imN45) / 255.0
im_arrN46 = np.array(imN46) / 255.0
im_arrN47 = np.array(imN47) / 255.0
im_arrN48 = np.array(imN48) / 255.0
im_arrN49 = np.array(imN49) / 255.0
im_arrN50 = np.array(imN50) / 255.0
im_arrN51 = np.array(imN51) / 255.0
im_arrN52 = np.array(imN52) / 255.0
im_arrN53 = np.array(imN53) / 255.0
im_arrN54 = np.array(imN54) / 255.0
im_arrN55 = np.array(imN55) / 255.0
im_arrN56 = np.array(imN56) / 255.0
im_arrN57 = np.array(imN57) / 255.0
im_arrN58 = np.array(imN58) / 255.0
im_arrN59 = np.array(imN59) / 255.0
im_arrN60 = np.array(imN60) / 255.0
im_arrN61 = np.array(imN61) / 255.0
im_arrN62 = np.array(imN62) / 255.0
im_arrN63 = np.array(imN63) / 255.0
im_arrN64 = np.array(imN64) / 255.0
im_arrN65 = np.array(imN65) / 255.0
im_arrN66 = np.array(imN66) / 255.0
im_arrN67 = np.array(imN67) / 255.0
im_arrN68 = np.array(imN68) / 255.0
im_arrN69 = np.array(imN69) / 255.0
im_arrN70 = np.array(imN70) / 255.0
im_arrN71 = np.array(imN71) / 255.0
im_arrN72 = np.array(imN72) / 255.0
im_arrN73 = np.array(imN73) / 255.0
im_arrN74 = np.array(imN74) / 255.0
im_arrN75 = np.array(imN75) / 255.0
im_arrN76 = np.array(imN76) / 255.0
im_arrN77 = np.array(imN77) / 255.0
im_arrN78 = np.array(imN78) / 255.0
im_arrN79 = np.array(imN79) / 255.0
im_arrN80 = np.array(imN80) / 255.0
im_arrN81 = np.array(imN81) / 255.0
im_arrN82 = np.array(imN82) / 255.0
im_arrN83 = np.array(imN83) / 255.0
im_arrN84 = np.array(imN84) / 255.0
im_arrN85 = np.array(imN85) / 255.0
im_arrN86 = np.array(imN86) / 255.0
im_arrN87 = np.array(imN87) / 255.0
im_arrN88 = np.array(imN88) / 255.0
im_arrN89 = np.array(imN89) / 255.0
im_arrN90 = np.array(imN90) / 255.0
im_arrN91 = np.array(imN91) / 255.0
im_arrN92 = np.array(imN92) / 255.0
im_arrN93 = np.array(imN93) / 255.0
im_arrN94 = np.array(imN94) / 255.0
im_arrN95 = np.array(imN95) / 255.0
im_arrN96 = np.array(imN96) / 255.0
im_arrN97 = np.array(imN97) / 255.0
im_arrN98 = np.array(imN98) / 255.0
im_arrN99 = np.array(imN99) / 255.0
im_arrN100 = np.array(imN100) / 255.0
im_arrN101 = np.array(imN101) / 255.0
im_arrN102 = np.array(imN102) / 255.0
im_arrN103 = np.array(imN103) / 255.0
im_arrN104 = np.array(imN104) / 255.0
im_arrN105 = np.array(imN105) / 255.0
im_arrN106 = np.array(imN106) / 255.0
im_arrN107 = np.array(imN107) / 255.0
im_arrN108 = np.array(imN108) / 255.0
im_arrN109 = np.array(imN109) / 255.0
im_arrN110 = np.array(imN110) / 255.0
im_arrN111 = np.array(imN111) / 255.0
im_arrN112 = np.array(imN112) / 255.0
im_arrN113 = np.array(imN113) / 255.0
im_arrN114 = np.array(imN114) / 255.0
im_arrN115 = np.array(imN115) / 255.0
im_arrN116 = np.array(imN116) / 255.0
im_arrN117 = np.array(imN117) / 255.0
im_arrN118 = np.array(imN118) / 255.0
im_arrN119 = np.array(imN119) / 255.0
im_arrN120 = np.array(imN120) / 255.0
im_arrN121 = np.array(imN121) / 255.0
im_arrN122 = np.array(imN122) / 255.0
im_arrN123 = np.array(imN123) / 255.0
im_arrN124 = np.array(imN124) / 255.0
im_arrN125 = np.array(imN125) / 255.0
im_arrN126 = np.array(imN126) / 255.0
im_arrN127 = np.array(imN127) / 255.0
im_arrN128 = np.array(imN128) / 255.0
im_arrN129 = np.array(imN129) / 255.0
im_arrN130 = np.array(imN130) / 255.0
im_arrN131 = np.array(imN131) / 255.0
im_arrN132 = np.array(imN132) / 255.0
im_arrN133 = np.array(imN133) / 255.0
im_arrN134 = np.array(imN134) / 255.0
im_arrN135 = np.array(imN135) / 255.0
im_arrN136 = np.array(imN136) / 255.0
im_arrN137 = np.array(imN137) / 255.0
im_arrN138 = np.array(imN138) / 255.0
im_arrN139 = np.array(imN139) / 255.0
im_arrN140 = np.array(imN140) / 255.0
im_arrN141 = np.array(imN141) / 255.0
im_arrN142 = np.array(imN142) / 255.0
im_arrN143 = np.array(imN143) / 255.0
im_arrN144 = np.array(imN144) / 255.0
im_arrN145 = np.array(imN145) / 255.0
im_arrN146 = np.array(imN146) / 255.0
im_arrN147 = np.array(imN147) / 255.0
im_arrN148 = np.array(imN148) / 255.0
im_arrN149 = np.array(imN149) / 255.0
im_arrN150 = np.array(imN150) / 255.0
im_arrN151 = np.array(imN151) / 255.0
im_arrN152 = np.array(imN152) / 255.0
im_arrN153 = np.array(imN153) / 255.0
im_arrN154 = np.array(imN154) / 255.0
im_arrN155 = np.array(imN155) / 255.0
im_arrN156 = np.array(imN156) / 255.0
im_arrN157 = np.array(imN157) / 255.0
im_arrN158 = np.array(imN158) / 255.0
im_arrN159 = np.array(imN159) / 255.0
im_arrN160 = np.array(imN160) / 255.0
im_arrN161 = np.array(imN161) / 255.0
im_arrN162 = np.array(imN162) / 255.0
im_arrN163 = np.array(imN163) / 255.0
im_arrN164 = np.array(imN164) / 255.0
im_arrN165 = np.array(imN165) / 255.0
im_arrN166 = np.array(imN166) / 255.0
im_arrN167 = np.array(imN167) / 255.0
im_arrN168 = np.array(imN168) / 255.0
im_arrN169 = np.array(imN169) / 255.0
im_arrN170 = np.array(imN170) / 255.0
im_arrN171 = np.array(imN171) / 255.0
im_arrN172 = np.array(imN172) / 255.0
im_arrN173 = np.array(imN173) / 255.0
im_arrN174 = np.array(imN174) / 255.0
im_arrN175 = np.array(imN175) / 255.0
im_arrN176 = np.array(imN176) / 255.0
im_arrN177 = np.array(imN177) / 255.0
im_arrN178 = np.array(imN178) / 255.0
im_arrN179 = np.array(imN179) / 255.0
im_arrN180 = np.array(imN180) / 255.0
im_arrN181 = np.array(imN181) / 255.0
im_arrN182 = np.array(imN182) / 255.0
im_arrN183 = np.array(imN183) / 255.0
im_arrN184 = np.array(imN184) / 255.0
im_arrN185 = np.array(imN185) / 255.0
im_arrN186 = np.array(imN186) / 255.0
im_arrN187 = np.array(imN187) / 255.0
im_arrN188 = np.array(imN188) / 255.0
im_arrN189 = np.array(imN189) / 255.0
im_arrN190 = np.array(imN190) / 255.0
im_arrN191 = np.array(imN191) / 255.0
im_arrN192 = np.array(imN192) / 255.0
im_arrN193 = np.array(imN193) / 255.0
im_arrN194 = np.array(imN194) / 255.0
im_arrN195 = np.array(imN195) / 255.0
im_arrN196 = np.array(imN196) / 255.0
im_arrN197 = np.array(imN197) / 255.0
im_arrN198 = np.array(imN198) / 255.0
im_arrN199 = np.array(imN199) / 255.0
im_arrN200 = np.array(imN200) / 255.0
im_arrA1 = np.array(imA1) / 255.0
im_arrA2 = np.array(imA2) / 255.0
im_arrA3 = np.array(imA3) / 255.0
im_arrA4 = np.array(imA4) / 255.0
im_arrA5 = np.array(imA5) / 255.0
im_arrA6 = np.array(imA6) / 255.0
im_arrA7 = np.array(imA7) / 255.0
im_arrA8 = np.array(imA8) / 255.0
im_arrA9 = np.array(imA9) / 255.0
im_arrA10 = np.array(imA10) / 255.0
im_arrA11 = np.array(imA11) / 255.0
im_arrA12 = np.array(imA12) / 255.0
im_arrA13 = np.array(imA13) / 255.0
im_arrA14 = np.array(imA14) / 255.0
im_arrA15 = np.array(imA15) / 255.0
im_arrA16 = np.array(imA16) / 255.0
im_arrA17 = np.array(imA17) / 255.0
im_arrA18 = np.array(imA18) / 255.0
im_arrA19 = np.array(imA19) / 255.0
im_arrA20 = np.array(imA20) / 255.0
im_arrA21 = np.array(imA21) / 255.0
im_arrA22 = np.array(imA22) / 255.0
im_arrA23 = np.array(imA23) / 255.0
im_arrA24 = np.array(imA24) / 255.0
im_arrA25 = np.array(imA25) / 255.0
im_arrA26 = np.array(imA26) / 255.0
im_arrA27 = np.array(imA27) / 255.0
im_arrA28 = np.array(imA28) / 255.0
im_arrA29 = np.array(imA29) / 255.0
im_arrA30 = np.array(imA30) / 255.0
im_arrA31 = np.array(imA31) / 255.0
im_arrA32 = np.array(imA32) / 255.0
im_arrA33 = np.array(imA33) / 255.0
im_arrA34 = np.array(imA34) / 255.0
im_arrA35 = np.array(imA35) / 255.0
im_arrA36 = np.array(imA36) / 255.0
im_arrA37 = np.array(imA37) / 255.0
im_arrA38 = np.array(imA38) / 255.0
im_arrA39 = np.array(imA39) / 255.0
im_arrA40 = np.array(imA40) / 255.0
im_arrA41 = np.array(imA41) / 255.0
im_arrA42 = np.array(imA42) / 255.0
im_arrA43 = np.array(imA43) / 255.0
im_arrA44 = np.array(imA44) / 255.0
im_arrA45 = np.array(imA45) / 255.0
im_arrA46 = np.array(imA46) / 255.0
im_arrA47 = np.array(imA47) / 255.0
im_arrA48 = np.array(imA48) / 255.0
im_arrA49 = np.array(imA49) / 255.0
im_arrA50 = np.array(imA50) / 255.0
im_arrA51 = np.array(imA51) / 255.0
im_arrA52 = np.array(imA52) / 255.0
im_arrA53 = np.array(imA53) / 255.0
im_arrA54 = np.array(imA54) / 255.0
im_arrA55 = np.array(imA55) / 255.0
im_arrA56 = np.array(imA56) / 255.0
im_arrA57 = np.array(imA57) / 255.0
im_arrA58 = np.array(imA58) / 255.0
im_arrA59 = np.array(imA59) / 255.0
im_arrA60 = np.array(imA60) / 255.0
im_arrA61 = np.array(imA61) / 255.0
im_arrA62 = np.array(imA62) / 255.0
im_arrA63 = np.array(imA63) / 255.0
im_arrA64 = np.array(imA64) / 255.0
im_arrA65 = np.array(imA65) / 255.0
im_arrA66 = np.array(imA66) / 255.0
im_arrA67 = np.array(imA67) / 255.0
im_arrA68 = np.array(imA68) / 255.0
im_arrA69 = np.array(imA69) / 255.0
im_arrA70 = np.array(imA70) / 255.0
im_arrA71 = np.array(imA71) / 255.0
im_arrA72 = np.array(imA72) / 255.0
im_arrA73 = np.array(imA73) / 255.0
im_arrA74 = np.array(imA74) / 255.0
im_arrA75 = np.array(imA75) / 255.0
im_arrA76 = np.array(imA76) / 255.0
im_arrA77 = np.array(imA77) / 255.0
im_arrA78 = np.array(imA78) / 255.0
im_arrA79 = np.array(imA79) / 255.0
im_arrA80 = np.array(imA80) / 255.0
im_arrA81 = np.array(imA81) / 255.0
im_arrA82 = np.array(imA82) / 255.0
im_arrA83 = np.array(imA83) / 255.0
im_arrA84 = np.array(imA84) / 255.0
im_arrA85 = np.array(imA85) / 255.0
im_arrA86 = np.array(imA86) / 255.0
im_arrA87 = np.array(imA87) / 255.0
im_arrA88 = np.array(imA88) / 255.0
im_arrA89 = np.array(imA89) / 255.0
im_arrA90 = np.array(imA90) / 255.0
im_arrA91 = np.array(imA91) / 255.0
im_arrA92 = np.array(imA92) / 255.0
im_arrA93 = np.array(imA93) / 255.0
im_arrA94 = np.array(imA94) / 255.0
im_arrA95 = np.array(imA95) / 255.0
im_arrA96 = np.array(imA96) / 255.0
im_arrA97 = np.array(imA97) / 255.0
im_arrA98 = np.array(imA98) / 255.0
im_arrA99 = np.array(imA99) / 255.0
im_arrA100 = np.array(imA100) / 255.0
im_arrA101 = np.array(imA101) / 255.0
im_arrA102 = np.array(imA102) / 255.0
im_arrA103 = np.array(imA103) / 255.0
im_arrA104 = np.array(imA104) / 255.0
im_arrA105 = np.array(imA105) / 255.0
im_arrA106 = np.array(imA106) / 255.0
im_arrA107 = np.array(imA107) / 255.0
im_arrA108 = np.array(imA108) / 255.0
im_arrA109 = np.array(imA109) / 255.0
im_arrA110 = np.array(imA110) / 255.0
im_arrA111 = np.array(imA111) / 255.0
im_arrA112 = np.array(imA112) / 255.0
im_arrA113 = np.array(imA113) / 255.0
im_arrA114 = np.array(imA114) / 255.0
im_arrA115 = np.array(imA115) / 255.0
im_arrA116 = np.array(imA116) / 255.0
im_arrA117 = np.array(imA117) / 255.0
im_arrA118 = np.array(imA118) / 255.0
im_arrA119 = np.array(imA119) / 255.0
im_arrA120 = np.array(imA120) / 255.0
im_arrA121 = np.array(imA121) / 255.0
im_arrA122 = np.array(imA122) / 255.0
im_arrA123 = np.array(imA123) / 255.0
im_arrA124 = np.array(imA124) / 255.0
im_arrA125 = np.array(imA125) / 255.0
im_arrA126 = np.array(imA126) / 255.0
im_arrA127 = np.array(imA127) / 255.0
im_arrA128 = np.array(imA128) / 255.0
im_arrA129 = np.array(imA129) / 255.0
im_arrA130 = np.array(imA130) / 255.0
im_arrA131 = np.array(imA131) / 255.0
im_arrA132 = np.array(imA132) / 255.0
im_arrA133 = np.array(imA133) / 255.0
im_arrA134 = np.array(imA134) / 255.0
im_arrA135 = np.array(imA135) / 255.0
im_arrA136 = np.array(imA136) / 255.0
im_arrA137 = np.array(imA137) / 255.0
im_arrA138 = np.array(imA138) / 255.0
im_arrA139 = np.array(imA139) / 255.0
im_arrA140 = np.array(imA140) / 255.0
im_arrA141 = np.array(imA141) / 255.0
im_arrA142 = np.array(imA142) / 255.0
im_arrA143 = np.array(imA143) / 255.0
im_arrA144 = np.array(imA144) / 255.0
im_arrA145 = np.array(imA145) / 255.0
im_arrA146 = np.array(imA146) / 255.0
im_arrA147 = np.array(imA147) / 255.0
im_arrA148 = np.array(imA148) / 255.0
im_arrA149 = np.array(imA149) / 255.0
im_arrA150 = np.array(imA150) / 255.0
im_arrA151 = np.array(imA151) / 255.0
im_arrA152 = np.array(imA152) / 255.0
im_arrA153 = np.array(imA153) / 255.0
im_arrA154 = np.array(imA154) / 255.0
im_arrA155 = np.array(imA155) / 255.0
im_arrA156 = np.array(imA156) / 255.0
im_arrA157 = np.array(imA157) / 255.0
im_arrA158 = np.array(imA158) / 255.0
im_arrA159 = np.array(imA159) / 255.0
im_arrA160 = np.array(imA160) / 255.0
im_arrA161 = np.array(imA161) / 255.0
im_arrA162 = np.array(imA162) / 255.0
im_arrA163 = np.array(imA163) / 255.0
im_arrA164 = np.array(imA164) / 255.0
im_arrA165 = np.array(imA165) / 255.0
im_arrA166 = np.array(imA166) / 255.0
im_arrA167 = np.array(imA167) / 255.0
im_arrA168 = np.array(imA168) / 255.0
im_arrA169 = np.array(imA169) / 255.0
im_arrA170 = np.array(imA170) / 255.0
im_arrA171 = np.array(imA171) / 255.0
im_arrA172 = np.array(imA172) / 255.0
im_arrA173 = np.array(imA173) / 255.0
im_arrA174 = np.array(imA174) / 255.0
im_arrA175 = np.array(imA175) / 255.0
im_arrA176 = np.array(imA176) / 255.0
im_arrA177 = np.array(imA177) / 255.0
im_arrA178 = np.array(imA178) / 255.0
im_arrA179 = np.array(imA179) / 255.0
im_arrA180 = np.array(imA180) / 255.0
im_arrA181 = np.array(imA181) / 255.0
im_arrA182 = np.array(imA182) / 255.0
im_arrA183 = np.array(imA183) / 255.0
im_arrA184 = np.array(imA184) / 255.0
im_arrA185 = np.array(imA185) / 255.0
im_arrA186 = np.array(imA186) / 255.0
im_arrA187 = np.array(imA187) / 255.0
im_arrA188 = np.array(imA188) / 255.0
im_arrA189 = np.array(imA189) / 255.0
im_arrA190 = np.array(imA190) / 255.0
im_arrA191 = np.array(imA191) / 255.0
im_arrA192 = np.array(imA192) / 255.0
im_arrA193 = np.array(imA193) / 255.0
im_arrA194 = np.array(imA194) / 255.0
im_arrA195 = np.array(imA195) / 255.0
im_arrA196 = np.array(imA196) / 255.0
im_arrA197 = np.array(imA197) / 255.0
im_arrA198 = np.array(imA198) / 255.0
im_arrA199 = np.array(imA199) / 255.0
im_arrA200 = np.array(imA200) / 255.0

#endregion

#region
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
image154 = im_arr154.flatten()
image155 = im_arr155.flatten()
image156 = im_arr156.flatten()
image157 = im_arr157.flatten()
image158 = im_arr158.flatten()
image159 = im_arr159.flatten()
image160 = im_arr160.flatten()
image161 = im_arr161.flatten()
image162 = im_arr162.flatten()
image163 = im_arr163.flatten()
image164 = im_arr164.flatten()
image165 = im_arr165.flatten()
image166 = im_arr166.flatten()
image167 = im_arr167.flatten()
image168 = im_arr168.flatten()
image169 = im_arr169.flatten()
image170 = im_arr170.flatten()
image171 = im_arr171.flatten()
image172 = im_arr172.flatten()
image173 = im_arr173.flatten()
image174 = im_arr174.flatten()
image175 = im_arr175.flatten()
image176 = im_arr176.flatten()
image177 = im_arr177.flatten()
image178 = im_arr178.flatten()
image179 = im_arr179.flatten()
image180 = im_arr180.flatten()
image181 = im_arr181.flatten()
image182 = im_arr182.flatten()
image183 = im_arr183.flatten()
image184 = im_arr184.flatten()
image185 = im_arr185.flatten()
image186 = im_arr186.flatten()
image187 = im_arr187.flatten()
image188 = im_arr188.flatten()
image189 = im_arr189.flatten()
image190 = im_arr190.flatten()
image191 = im_arr191.flatten()
image192 = im_arr192.flatten()
image193 = im_arr193.flatten()
image194 = im_arr194.flatten()
image195 = im_arr195.flatten()
image196 = im_arr196.flatten()
image197 = im_arr197.flatten()
image198 = im_arr198.flatten()
image199 = im_arr199.flatten()
image200 = im_arr200.flatten()
#NOT HAPPY
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
imageN13 = im_arrN13.flatten()
imageN14 = im_arrN14.flatten()
imageN15 = im_arrN15.flatten()
imageN16 = im_arrN16.flatten()
imageN17 = im_arrN17.flatten()
imageN18 = im_arrN18.flatten()
imageN19 = im_arrN19.flatten()
imageN20 = im_arrN20.flatten()
imageN21 = im_arrN21.flatten()
imageN22 = im_arrN22.flatten()
imageN23 = im_arrN23.flatten()
imageN24 = im_arrN24.flatten()
imageN25 = im_arrN25.flatten()
imageN26 = im_arrN26.flatten()
imageN27 = im_arrN27.flatten()
imageN28 = im_arrN28.flatten()
imageN29 = im_arrN29.flatten()
imageN30 = im_arrN30.flatten()
imageN31 = im_arrN31.flatten()
imageN32 = im_arrN32.flatten()
imageN33 = im_arrN33.flatten()
imageN34 = im_arrN34.flatten()
imageN35 = im_arrN35.flatten()
imageN36 = im_arrN36.flatten()
imageN37 = im_arrN37.flatten()
imageN38 = im_arrN38.flatten()
imageN39 = im_arrN39.flatten()
imageN40 = im_arrN40.flatten()
imageN41 = im_arrN41.flatten()
imageN42 = im_arrN42.flatten()
imageN43 = im_arrN43.flatten()
imageN44 = im_arrN44.flatten()
imageN45 = im_arrN45.flatten()
imageN46 = im_arrN46.flatten()
imageN47 = im_arrN47.flatten()
imageN48 = im_arrN48.flatten()
imageN49 = im_arrN49.flatten()
imageN50 = im_arrN50.flatten()
imageN51 = im_arrN51.flatten()
imageN52 = im_arrN52.flatten()
imageN53 = im_arrN53.flatten()
imageN54 = im_arrN54.flatten()
imageN55 = im_arrN55.flatten()
imageN56 = im_arrN56.flatten()
imageN57 = im_arrN57.flatten()
imageN58 = im_arrN58.flatten()
imageN59 = im_arrN59.flatten()
imageN60 = im_arrN60.flatten()
imageN61 = im_arrN61.flatten()
imageN62 = im_arrN62.flatten()
imageN63 = im_arrN63.flatten()
imageN64 = im_arrN64.flatten()
imageN65 = im_arrN65.flatten()
imageN66 = im_arrN66.flatten()
imageN67 = im_arrN67.flatten()
imageN68 = im_arrN68.flatten()
imageN69 = im_arrN69.flatten()
imageN70 = im_arrN70.flatten()
imageN71 = im_arrN71.flatten()
imageN72 = im_arrN72.flatten()
imageN73 = im_arrN73.flatten()
imageN74 = im_arrN74.flatten()
imageN75 = im_arrN75.flatten()
imageN76 = im_arrN76.flatten()
imageN77 = im_arrN77.flatten()
imageN78 = im_arrN78.flatten()
imageN79 = im_arrN79.flatten()
imageN80 = im_arrN80.flatten()
imageN81 = im_arrN81.flatten()
imageN82 = im_arrN82.flatten()
imageN83 = im_arrN83.flatten()
imageN84 = im_arrN84.flatten()
imageN85 = im_arrN85.flatten()
imageN86 = im_arrN86.flatten()
imageN87 = im_arrN87.flatten()
imageN88 = im_arrN88.flatten()
imageN89 = im_arrN89.flatten()
imageN90 = im_arrN90.flatten()
imageN91 = im_arrN91.flatten()
imageN92 = im_arrN92.flatten()
imageN93 = im_arrN93.flatten()
imageN94 = im_arrN94.flatten()
imageN95 = im_arrN95.flatten()
imageN96 = im_arrN96.flatten()
imageN97 = im_arrN97.flatten()
imageN98 = im_arrN98.flatten()
imageN99 = im_arrN99.flatten()
imageN100 = im_arrN100.flatten()
imageN101 = im_arrN101.flatten()
imageN102 = im_arrN102.flatten()
imageN103 = im_arrN103.flatten()
imageN104 = im_arrN104.flatten()
imageN105 = im_arrN105.flatten()
imageN106 = im_arrN106.flatten()
imageN107 = im_arrN107.flatten()
imageN108 = im_arrN108.flatten()
imageN109 = im_arrN109.flatten()
imageN110 = im_arrN110.flatten()
imageN111 = im_arrN111.flatten()
imageN112 = im_arrN112.flatten()
imageN113 = im_arrN113.flatten()
imageN114 = im_arrN114.flatten()
imageN115 = im_arrN115.flatten()
imageN116 = im_arrN116.flatten()
imageN117 = im_arrN117.flatten()
imageN118 = im_arrN118.flatten()
imageN119 = im_arrN119.flatten()
imageN120 = im_arrN120.flatten()
imageN121 = im_arrN121.flatten()
imageN122 = im_arrN122.flatten()
imageN123 = im_arrN123.flatten()
imageN124 = im_arrN124.flatten()
imageN125 = im_arrN125.flatten()
imageN126 = im_arrN126.flatten()
imageN127 = im_arrN127.flatten()
imageN128 = im_arrN128.flatten()
imageN129 = im_arrN129.flatten()
imageN130 = im_arrN130.flatten()
imageN131 = im_arrN131.flatten()
imageN132 = im_arrN132.flatten()
imageN133 = im_arrN133.flatten()
imageN134 = im_arrN134.flatten()
imageN135 = im_arrN135.flatten()
imageN136 = im_arrN136.flatten()
imageN137 = im_arrN137.flatten()
imageN138 = im_arrN138.flatten()
imageN139 = im_arrN139.flatten()
imageN140 = im_arrN140.flatten()
imageN141 = im_arrN141.flatten()
imageN142 = im_arrN142.flatten()
imageN143 = im_arrN143.flatten()
imageN144 = im_arrN144.flatten()
imageN145 = im_arrN145.flatten()
imageN146 = im_arrN146.flatten()
imageN147 = im_arrN147.flatten()
imageN148 = im_arrN148.flatten()
imageN149 = im_arrN149.flatten()
imageN150 = im_arrN150.flatten()
imageN151 = im_arrN151.flatten()
imageN152 = im_arrN152.flatten()
imageN153 = im_arrN153.flatten()
imageN154 = im_arrN154.flatten()
imageN155 = im_arrN155.flatten()
imageN156 = im_arrN156.flatten()
imageN157 = im_arrN157.flatten()
imageN158 = im_arrN158.flatten()
imageN159 = im_arrN159.flatten()
imageN160 = im_arrN160.flatten()
imageN161 = im_arrN161.flatten()
imageN162 = im_arrN162.flatten()
imageN163 = im_arrN163.flatten()
imageN164 = im_arrN164.flatten()
imageN165 = im_arrN165.flatten()
imageN166 = im_arrN166.flatten()
imageN167 = im_arrN167.flatten()
imageN168 = im_arrN168.flatten()
imageN169 = im_arrN169.flatten()
imageN170 = im_arrN170.flatten()
imageN171 = im_arrN171.flatten()
imageN172 = im_arrN172.flatten()
imageN173 = im_arrN173.flatten()
imageN174 = im_arrN174.flatten()
imageN175 = im_arrN175.flatten()
imageN176 = im_arrN176.flatten()
imageN177 = im_arrN177.flatten()
imageN178 = im_arrN178.flatten()
imageN179 = im_arrN179.flatten()
imageN180 = im_arrN180.flatten()
imageN181 = im_arrN181.flatten()
imageN182 = im_arrN182.flatten()
imageN183 = im_arrN183.flatten()
imageN184 = im_arrN184.flatten()
imageN185 = im_arrN185.flatten()
imageN186 = im_arrN186.flatten()
imageN187 = im_arrN187.flatten()
imageN188 = im_arrN188.flatten()
imageN189 = im_arrN189.flatten()
imageN190 = im_arrN190.flatten()
imageN191 = im_arrN191.flatten()
imageN192 = im_arrN192.flatten()
imageN193 = im_arrN193.flatten()
imageN194 = im_arrN194.flatten()
imageN195 = im_arrN195.flatten()
imageN196 = im_arrN196.flatten()
imageN197 = im_arrN197.flatten()
imageN198 = im_arrN198.flatten()
imageN199 = im_arrN199.flatten()
imageN200 = im_arrN200.flatten()
imageA1 = im_arrA1.flatten()
imageA2 = im_arrA2.flatten()
imageA3 = im_arrA3.flatten()
imageA4 = im_arrA4.flatten()
imageA5 = im_arrA5.flatten()
imageA6 = im_arrA6.flatten()
imageA7 = im_arrA7.flatten()
imageA8 = im_arrA8.flatten()
imageA9 = im_arrA9.flatten()
imageA10 = im_arrA10.flatten()
imageA11 = im_arrA11.flatten()
imageA12 = im_arrA12.flatten()
imageA13 = im_arrA13.flatten()
imageA14 = im_arrA14.flatten()
imageA15 = im_arrA15.flatten()
imageA16 = im_arrA16.flatten()
imageA17 = im_arrA17.flatten()
imageA18 = im_arrA18.flatten()
imageA19 = im_arrA19.flatten()
imageA20 = im_arrA20.flatten()
imageA21 = im_arrA21.flatten()
imageA22 = im_arrA22.flatten()
imageA23 = im_arrA23.flatten()
imageA24 = im_arrA24.flatten()
imageA25 = im_arrA25.flatten()
imageA26 = im_arrA26.flatten()
imageA27 = im_arrA27.flatten()
imageA28 = im_arrA28.flatten()
imageA29 = im_arrA29.flatten()
imageA30 = im_arrA30.flatten()
imageA31 = im_arrA31.flatten()
imageA32 = im_arrA32.flatten()
imageA33 = im_arrA33.flatten()
imageA34 = im_arrA34.flatten()
imageA35 = im_arrA35.flatten()
imageA36 = im_arrA36.flatten()
imageA37 = im_arrA37.flatten()
imageA38 = im_arrA38.flatten()
imageA39 = im_arrA39.flatten()
imageA40 = im_arrA40.flatten()
imageA41 = im_arrA41.flatten()
imageA42 = im_arrA42.flatten()
imageA43 = im_arrA43.flatten()
imageA44 = im_arrA44.flatten()
imageA45 = im_arrA45.flatten()
imageA46 = im_arrA46.flatten()
imageA47 = im_arrA47.flatten()
imageA48 = im_arrA48.flatten()
imageA49 = im_arrA49.flatten()
imageA50 = im_arrA50.flatten()
imageA51 = im_arrA51.flatten()
imageA52 = im_arrA52.flatten()
imageA53 = im_arrA53.flatten()
imageA54 = im_arrA54.flatten()
imageA55 = im_arrA55.flatten()
imageA56 = im_arrA56.flatten()
imageA57 = im_arrA57.flatten()
imageA58 = im_arrA58.flatten()
imageA59 = im_arrA59.flatten()
imageA60 = im_arrA60.flatten()
imageA61 = im_arrA61.flatten()
imageA62 = im_arrA62.flatten()
imageA63 = im_arrA63.flatten()
imageA64 = im_arrA64.flatten()
imageA65 = im_arrA65.flatten()
imageA66 = im_arrA66.flatten()
imageA67 = im_arrA67.flatten()
imageA68 = im_arrA68.flatten()
imageA69 = im_arrA69.flatten()
imageA70 = im_arrA70.flatten()
imageA71 = im_arrA71.flatten()
imageA72 = im_arrA72.flatten()
imageA73 = im_arrA73.flatten()
imageA74 = im_arrA74.flatten()
imageA75 = im_arrA75.flatten()
imageA76 = im_arrA76.flatten()
imageA77 = im_arrA77.flatten()
imageA78 = im_arrA78.flatten()
imageA79 = im_arrA79.flatten()
imageA80 = im_arrA80.flatten()
imageA81 = im_arrA81.flatten()
imageA82 = im_arrA82.flatten()
imageA83 = im_arrA83.flatten()
imageA84 = im_arrA84.flatten()
imageA85 = im_arrA85.flatten()
imageA86 = im_arrA86.flatten()
imageA87 = im_arrA87.flatten()
imageA88 = im_arrA88.flatten()
imageA89 = im_arrA89.flatten()
imageA90 = im_arrA90.flatten()
imageA91 = im_arrA91.flatten()
imageA92 = im_arrA92.flatten()
imageA93 = im_arrA93.flatten()
imageA94 = im_arrA94.flatten()
imageA95 = im_arrA95.flatten()
imageA96 = im_arrA96.flatten()
imageA97 = im_arrA97.flatten()
imageA98 = im_arrA98.flatten()
imageA99 = im_arrA99.flatten()
imageA100 = im_arrA100.flatten()
imageA101 = im_arrA101.flatten()
imageA102 = im_arrA102.flatten()
imageA103 = im_arrA103.flatten()
imageA104 = im_arrA104.flatten()
imageA105 = im_arrA105.flatten()
imageA106 = im_arrA106.flatten()
imageA107 = im_arrA107.flatten()
imageA108 = im_arrA108.flatten()
imageA109 = im_arrA109.flatten()
imageA110 = im_arrA110.flatten()
imageA111 = im_arrA111.flatten()
imageA112 = im_arrA112.flatten()
imageA113 = im_arrA113.flatten()
imageA114 = im_arrA114.flatten()
imageA115 = im_arrA115.flatten()
imageA116 = im_arrA116.flatten()
imageA117 = im_arrA117.flatten()
imageA118 = im_arrA118.flatten()
imageA119 = im_arrA119.flatten()
imageA120 = im_arrA120.flatten()
imageA121 = im_arrA121.flatten()
imageA122 = im_arrA122.flatten()
imageA123 = im_arrA123.flatten()
imageA124 = im_arrA124.flatten()
imageA125 = im_arrA125.flatten()
imageA126 = im_arrA126.flatten()
imageA127 = im_arrA127.flatten()
imageA128 = im_arrA128.flatten()
imageA129 = im_arrA129.flatten()
imageA130 = im_arrA130.flatten()
imageA131 = im_arrA131.flatten()
imageA132 = im_arrA132.flatten()
imageA133 = im_arrA133.flatten()
imageA134 = im_arrA134.flatten()
imageA135 = im_arrA135.flatten()
imageA136 = im_arrA136.flatten()
imageA137 = im_arrA137.flatten()
imageA138 = im_arrA138.flatten()
imageA139 = im_arrA139.flatten()
imageA140 = im_arrA140.flatten()
imageA141 = im_arrA141.flatten()
imageA142 = im_arrA142.flatten()
imageA143 = im_arrA143.flatten()
imageA144 = im_arrA144.flatten()
imageA145 = im_arrA145.flatten()
imageA146 = im_arrA146.flatten()
imageA147 = im_arrA147.flatten()
imageA148 = im_arrA148.flatten()
imageA149 = im_arrA149.flatten()
imageA150 = im_arrA150.flatten()
imageA151 = im_arrA151.flatten()
imageA152 = im_arrA152.flatten()
imageA153 = im_arrA153.flatten()
imageA154 = im_arrA154.flatten()
imageA155 = im_arrA155.flatten()
imageA156 = im_arrA156.flatten()
imageA157 = im_arrA157.flatten()
imageA158 = im_arrA158.flatten()
imageA159 = im_arrA159.flatten()
imageA160 = im_arrA160.flatten()
imageA161 = im_arrA161.flatten()
imageA162 = im_arrA162.flatten()
imageA163 = im_arrA163.flatten()
imageA164 = im_arrA164.flatten()
imageA165 = im_arrA165.flatten()
imageA166 = im_arrA166.flatten()
imageA167 = im_arrA167.flatten()
imageA168 = im_arrA168.flatten()
imageA169 = im_arrA169.flatten()
imageA170 = im_arrA170.flatten()
imageA171 = im_arrA171.flatten()
imageA172 = im_arrA172.flatten()
imageA173 = im_arrA173.flatten()
imageA174 = im_arrA174.flatten()
imageA175 = im_arrA175.flatten()
imageA176 = im_arrA176.flatten()
imageA177 = im_arrA177.flatten()
imageA178 = im_arrA178.flatten()
imageA179 = im_arrA179.flatten()
imageA180 = im_arrA180.flatten()
imageA181 = im_arrA181.flatten()
imageA182 = im_arrA182.flatten()
imageA183 = im_arrA183.flatten()
imageA184 = im_arrA184.flatten()
imageA185 = im_arrA185.flatten()
imageA186 = im_arrA186.flatten()
imageA187 = im_arrA187.flatten()
imageA188 = im_arrA188.flatten()
imageA189 = im_arrA189.flatten()
imageA190 = im_arrA190.flatten()
imageA191 = im_arrA191.flatten()
imageA192 = im_arrA192.flatten()
imageA193 = im_arrA193.flatten()
imageA194 = im_arrA194.flatten()
imageA195 = im_arrA195.flatten()
imageA196 = im_arrA196.flatten()
imageA197 = im_arrA197.flatten()
imageA198 = im_arrA198.flatten()
imageA199 = im_arrA199.flatten()
imageA200 = im_arrA200.flatten()

#endregion


dataset = np.array([image1, image2, image3, image4, image5, image6, image7, image8, image9, image10, image11,
                    image12, image13, image14, image15, image16, image17, image18, image19, image20, image21,
                    image22, image23, image24, image25, image26, image27, image28, image29, image30, image31,
                    image32, image33, image34, image35, image36, image37, image38, image39, image40, image41,
                    image42, image43, image44, image45, image46, image47, image48, image49, image50, image51,
                    image52, image53, image54, image55, image56, image57, image58, image59, image60, image61,
                    image62, image63, image64, image65, image66, image67, image68, image69, image70, image71,
                    image72, image73, image74, image75, image76, image77, image78, image79, image80, image81,
                    image82, image83, image84, image85, image86, image87, image88, image89, image90, image91,
                    image92, image93, image94, image95, image96, image97, image98, image99, image100, image101,
                    image102, image103, image104, image105, image106, image107, image108, image109, image110, image111,
                    image112, image113, image114, image115, image116, image117, image118, image119, image120, image121,
                    image122, image123, image124, image125, image126, image127, image128, image129, image130, image131,
                    image132, image133, image134, image135, image136, image137, image138, image139, image140, image141,
                    image142, image143, image144, image145, image146, image147, image148, image149, image150, image151,
                    image152, image153, image154, image155, image156, image157, image158, image159, image160, image161,
                    image162, image163, image164, image165, image166, image167, image168, image169, image170, image171,
                    image172, image173, image174, image175, image176, image177, image178, image179, image180, image181,
                    image182, image183, image184, image185, image186, image187, image188, image189, image190, image191,
                    image192, image193, image194, image195, image196, image197, image198, image199, image200,

                    imageN1, imageN2, imageN3, imageN4, imageN5, imageN6, imageN7, imageN8, imageN9, imageN10, imageN11,
                    imageN12, imageN13, imageN14, imageN15, imageN16, imageN17, imageN18, imageN19, imageN20,  imageN21,
                    imageN22, imageN23, imageN24, imageN25, imageN26, imageN27, imageN28, imageN29, imageN30, imageN31,
                    imageN32, imageN33, imageN34, imageN35, imageN36, imageN37, imageN38, imageN39, imageN40, imageN41,
                    imageN42, imageN43, imageN44, imageN45, imageN46, imageN47, imageN48, imageN49, imageN50, imageN51,
                    imageN52, imageN53, imageN54, imageN55, imageN56, imageN57, imageN58, imageN59, imageN60, imageN61,
                    imageN62, imageN63, imageN64, imageN65, imageN66, imageN67, imageN68, imageN69, imageN70, imageN71,
                    imageN72, imageN73, imageN74, imageN75, imageN76, imageN77, imageN78, imageN79, imageN80, imageN81,
                    imageN82, imageN83, imageN84, imageN85, imageN86, imageN87, imageN88, imageN89, imageN90, imageN91,
                    imageN92, imageN93, imageN94, imageN95, imageN96, imageN97, imageN98, imageN99, imageN100, imageN101,
                    imageN102, imageN103, imageN104, imageN105, imageN106, imageN107, imageN108, imageN109, imageN110, imageN111,
                    imageN112, imageN113, imageN114, imageN115, imageN116, imageN117, imageN118, imageN119, imageN120, imageN121,
                    imageN122, imageN123, imageN124, imageN125, imageN126, imageN127, imageN128, imageN129, imageN130, imageN131,
                    imageN132, imageN133, imageN134, imageN135, imageN136, imageN137, imageN138, imageN139, imageN140, imageN141,
                    imageN142, imageN143, imageN144, imageN145, imageN146, imageN147, imageN148, imageN149, imageN150, imageN151,
                    imageN152, imageN153, imageN154, imageN155, imageN156, imageN157, imageN158, imageN159, imageN160, imageN161,
                    imageN162, imageN163, imageN164, imageN165, imageN166, imageN167, imageN168, imageN169, imageN170, imageN171,
                    imageN172, imageN173, imageN174, imageN175, imageN176, imageN177, imageN178, imageN179, imageN180, imageN181,
                    imageN182, imageN183, imageN184, imageN185, imageN186, imageN187, imageN188, imageN189, imageN190, imageN191,
                    imageN192, imageN193, imageN194, imageN195, imageN196, imageN197, imageN198, imageN199, imageN200,


                    imageA1, imageA2, imageA3, imageA4, imageA5, imageA6, imageA7, imageA8, imageA9, imageA10, imageA11,
                    imageA12, imageA13, imageA14, imageA15, imageA16, imageA17, imageA18, imageA19, imageA20, imageA21,
                    imageA22, imageA23, imageA24, imageA25, imageA26, imageA27, imageA28, imageA29, imageA30, imageA31,
                    imageA32, imageA33, imageA34, imageA35, imageA36, imageA37, imageA38, imageA39, imageA40, imageA41,
                    imageA42, imageA43, imageA44, imageA45, imageA46, imageA47, imageA48, imageA49, imageA50, imageA51,
                    imageA52, imageA53, imageA54, imageA55, imageA56, imageA57, imageA58, imageA59, imageA60, imageA61,
                    imageA62, imageA63, imageA64, imageA65, imageA66, imageA67, imageA68, imageA69, imageA70, imageA71,
                    imageA72, imageA73, imageA74, imageA75, imageA76, imageA77, imageA78, imageA79, imageA80, imageA81,
                    imageA82, imageA83, imageA84, imageA85, imageA86, imageA87, imageA88, imageA89, imageA90, imageA91,
                    imageA92, imageA93, imageA94, imageA95, imageA96, imageA97, imageA98, imageA99, imageA100,imageA101,
                    imageA102, imageA103, imageA104, imageA105, imageA106, imageA107, imageA108, imageA109, imageA110,imageA111,
                    imageA112, imageA113, imageA114, imageA115, imageA116, imageA117, imageA118, imageA119, imageA120,imageA121,
                    imageA122, imageA123, imageA124, imageA125, imageA126, imageA127, imageA128, imageA129, imageA130,imageA131,
                    imageA132, imageA133, imageA134, imageA135, imageA136, imageA137, imageA138, imageA139, imageA140,imageA141,
                    imageA142, imageA143, imageA144, imageA145, imageA146, imageA147, imageA148, imageA149, imageA150,imageA151,
                    imageA152, imageA153, imageA154, imageA155, imageA156, imageA157, imageA158, imageA159, imageA160, imageA161,
                    imageA162, imageA163, imageA164, imageA165, imageA166, imageA167, imageA168, imageA169, imageA170, imageA171,
                    imageA172, imageA173, imageA174, imageA175, imageA176, imageA177, imageA178, imageA179, imageA180, imageA181,
                    imageA182, imageA183, imageA184, imageA185, imageA186, imageA187, imageA188, imageA189, imageA190, imageA191,
                    imageA192, imageA193, imageA194, imageA195, imageA196, imageA197, imageA198, imageA199, imageA200], dtype='float64')




dataset_expected_output = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,

                                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,

                                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
                                    ], dtype='float64')


L = np.array([
    2,
    2,
    1
], dtype='int32')


model = my_lib.create_MLP_model(L.shape[0], L.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))


flattened_Dataset = dataset.flatten()



alpha = 1
iteration = 10000

my_lib.train_MLP_Classification(
    model,
    L.shape[0],
    L.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
    flattened_Dataset.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    dataset_expected_output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    dataset.shape[0],
    dataset.shape[1],
    alpha,
    iteration
)



print("DATASET DE TRAIN:")

count = 0
bad = 0
error = 0.2

for inputs_k in dataset:
    output = my_lib.predict_MLP_Classification(
        model,
        L.shape[0],
        L.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        inputs_k.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    #print(output[0])
    #print(dataset_expected_output[count])
    if output[0] != dataset_expected_output[count]:
        if abs(output[0] - dataset_expected_output[count]) > error:
            bad = bad + 1
        count = count + 1



print(percentOfGoodPrediction(dataset.shape[0], bad), "% de bonne prédiction")
print(percentOfBadPrediction(dataset.shape[0], bad), "% de mauvaise prédiction")



my_lib.dispose_MLP(model)
