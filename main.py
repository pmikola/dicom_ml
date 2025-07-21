import copy
import os
import sys

import cv2
import numpy as np
import cv2 as cv
from skimage.feature import match_template, peak_local_max
from skimage.transform import rescale, resize, downscale_local_mean
from matplotlib import pyplot as plt
import matplotlib
from pydicom.data import get_testdata_file
import pydicom
from PIL import Image
import time
from PIL import Image, ImageEnhance
import scipy.signal

from model import HairSkinClassifier

matplotlib.use("TkAgg")

start = time.time()
def show_all_pic(images, disp):
    if disp == 1:
        fig = plt.figure(figsize=(5, 5))
        rows = 4
        columns = 4
        for i in range(len(images)):
            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(images[i].astype('uint8'))
            plt.axis('off')
        plt.tight_layout(h_pad=1, w_pad=1)
        plt.show()
    else:
        pass




def ClaheHSV(image, clip, grid_size):
    hsv = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid_size)
    result = clahe.apply(v)
    hsv = cv2.merge((h, s, result))
    equalized_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return equalized_img


def Enhance(images):
    img = []
    hist = []
    for i in range(len(images)):
        res = ClaheHSV(images[i], 5., (10, 10))
        norm = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX)
        img.append(norm)
    return img, hist


def image_avg(images):
    avg_image = images[0]
    for i in range(len(images)):
        if i == 0:
            pass
        else:
            alpha = 1.0 / (i + 1)
            beta = 1.0 - alpha
            avg_image = cv2.addWeighted(images[i], alpha, avg_image, beta, 0.0)
    return avg_image

####################### KERNELS
####################### # True Sharp
kernel_sharp = np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]])

####################### # Sharpen
kernel_sharp2 = np.array([[-1., -1., -1.],
                          [-1., 9., -1.],
                          [-1., -1., -1.]])
####################### # Identity
kernel_identity = np.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]])

####################### # Embossed (rate of color change)
kernel_emboss = np.array([[-1, -1, 0],
                          [-1, 0, 1],
                          [0, 1, 1]])

####################### # Edge detection
kernel_edge = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]])

####################### IMPORT DCM
base = 'C:\PRACA\Depi\Prg_prj\ml_dicom\dicom_ml'
pass_dicom1 = '2024-03-18-14-39-21_d5b95953-c78a-4cc4-b596-c65caa435990'
pass_dicom = pass_dicom1

filename = pydicom.data.data_manager.get_files(base, pass_dicom + '.dcm')[0]

x = pydicom.dcmread(filename, force=True)
model = HairSkinClassifier(disp=1)
x = model(x)
# show_all_pic([image_comb,image_comb],1)

sys.exit()


# im = Image.fromarray(image_downscaled.astype('uint8'))
# im.save(pass_dicom + '.png')


###### BLUR DETECTION START
def detect_blur_fft(image, size=60, thresh=10, vis=False):
    h = image.shape[0]
    w = image.shape[1]
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    fft = np.fft.fft2(image.astype('uint8'))
    fftShift = np.fft.fftshift(fft)
    if vis:
        magnitude = 20 * np.log(np.abs(fftShift))
        (fig, ax) = plt.subplots(1, 2, )
        ax[0].imshow(image.astype('uint8'), cmap="gray")
        ax[0].set_title("Input")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].imshow(magnitude.astype('uint8'), cmap="gray")
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        plt.show()
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    return mean, mean <= thresh


gray = cv2.cvtColor(image_downscaled, cv2.COLOR_RGB2GRAY)



end = time.time()

plt.show()

print("Evaluation time: ", end - start)
