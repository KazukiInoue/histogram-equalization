import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

import inspect


class Img:
    def __init__(self):
        self.image = []
        self.channels = []
    

def opencv_eqhist(from_dir, to_dir):

    files = os.listdir(from_dir)
    iteration = 1

    for file in files:
        bgr = Img()
        bgr.image = cv2.imread(from_dir+'/'+file, 1)
        bgr.channels = cv2.split(bgr.image)

        dst = Img()
        dst.channels = bgr.channels

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # cl1 = clahe.apply(img)
        for c in range(3):
            # dst.channels[c] = cv2.equalizeHist(bgr.channels[c])
            dst.channels[c] = clahe.apply(bgr.channels[c])

        dst.image = cv2.merge(bgr.channels)
        tmp_output_name = file.split(".")
        output_name = tmp_output_name[0]
        cv2.imwrite(to_dir+output_name+"_eqhist.png", dst.image)
        iteration = iteration + 1


def handmade_eqhist(input_img_path, to_dir):
    # -----------オリジナルの関数 - ---------

    # [[1,2],[3]].flatten()->[1,2,3]
    img = cv2.imread(input_img_path)

    bgr = cv2.split(img)

    for color in range(3):

        hist, bins = np.histogram(bgr[color].flatten(), 256, [0, 256])

        # cumsumで累積分布の算出 cumulative distribution
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()

        plt.title('before')
        plt.plot(cdf_normalized, color='b')
        plt.hist(bgr[color].flatten(), 255, [0, 255], color='r')
        plt.xlim([0, 255])
        plt.legend(('cdf_normalized', 'histogram'), loc='upper left')
        plt.show()

        # ----------ヒストグラムの平坦化----------
        # 最小値(0ではない)を探すために0にマスクをかける
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        img2 = cdf[bgr[color]]

        hist, bins = np.histogram(img2.flatten(), 256, [0, 255])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()

        plt.title('after')
        plt.plot(cdf_normalized, color='b')
        plt.plot(cdf_m, color='g')
        plt.hist(img2.flatten(), 256, [0, 255], color='r')
        plt.xlim([0, 255])
        plt.legend(('cdf_normalized', 'cdf_m', 'histogram'), loc='upper left')
        plt.show()

        cv2.imwrite(to_dir+"src_after_"+str(color)+".jpg", img2)


if __name__ == '__main__':
    opencv_eqhist('./input_images/', './output_images/')
    # handmade_eqhist(input_img_path="./input_images/1060_real_B.png",
    #                 to_dir="./output_images/")

