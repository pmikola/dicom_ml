
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

dsr = pydicom.dcmread(filename, force=True)
dsg = pydicom.dcmread(filename, force=True)
dsb = pydicom.dcmread(filename, force=True)


def getnshow_dicom(ds, rgb, disp):
    images = []
    for i in range(ds.pixel_array.shape[0]):
        if rgb == 'r':
            ds.pixel_array[i, :, :, 2] = 0
            ds.pixel_array[i, :, :, 1] = 0
        if rgb == 'b':
            ds.pixel_array[i, :, :, 0] = 0
            ds.pixel_array[i, :, :, 1] = 0
        if rgb == 'g':
            ds.pixel_array[i, :, :, 2] = 0
            ds.pixel_array[i, :, :, 0] = 0
        else:
            pass
        images.append(ds.pixel_array[i, :, :, :])
    show_all_pic(images, disp)
    return images

####################### PROCESSING
####################### # PXARR
disp = 0
imagesr = getnshow_dicom(dsr, 'r', disp)
imagesg = getnshow_dicom(dsg, 'g', disp)
imagesb = getnshow_dicom(dsb, 'b', disp)

####################### # SHARP
image_kernelr = []
image_kernelg = []
image_kernelb = []
for i in range(len(imagesr)):
    image_kernelr.append(cv2.filter2D(src=imagesr[i], ddepth=-1, kernel=kernel_sharp))
    image_kernelg.append(cv2.filter2D(src=imagesg[i], ddepth=-1, kernel=kernel_sharp))
    image_kernelb.append(cv2.filter2D(src=imagesb[i], ddepth=-1, kernel=kernel_sharp))

####################### # NORM
images_normr, histr = Enhance(image_kernelr)
images_normg, histg = Enhance(image_kernelg)
images_normb, histb = Enhance(image_kernelb)

####################### # 14xColor
imgcol = np.zeros((14, 972, 1296, 3))
imgcol[:, :, :, 0] = np.array(images_normr)[:, :, :, 0]
imgcol[:, :, :, 1] = np.array(images_normg)[:, :, :, 1]
imgcol[:, :, :, 2] = np.array(images_normb)[:, :, :, 2]

col_avg = image_avg(imgcol)
avg_imager = image_avg(images_normr)
avg_imageg = image_avg(images_normg)
avg_imageb = image_avg(images_normb)

norm_param = cv2.NORM_MINMAX
avg_normb = cv2.normalize(avg_imageb, None, 0, 255, norm_param)
avg_normg = cv2.normalize(avg_imageg, None, 0, 255, norm_param)
avg_normr = cv2.normalize(avg_imager, None, 0, 255, norm_param)
avg_images = [avg_normb, avg_normg, avg_normr]

# show_all_pic(avg_images,1)
####################### # 3X1 TO 1X3
image_comb = np.array(avg_imager)
image_comb[:, :, 0] = avg_imager[:, :, 0]
image_comb[:, :, 1] = avg_imageg[:, :, 1]
image_comb[:, :, 2] = avg_imageb[:, :, 2]

# show_all_pic([image_comb,image_comb],1)
plt.imshow(image_comb)
plt.show()
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
