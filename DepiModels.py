import copy

import numpy as np
import torch
from torch import nn
from transformers import ViTForImageClassification, AutoImageProcessor
import cv2
from matplotlib import pyplot as plt
from transformers import logging

logging.set_verbosity_error()
hair_color_model_dir = './hc_model'
skin_type_model_dir = './skin_type_model'

hc_model =  ViTForImageClassification.from_pretrained(hair_color_model_dir,local_files_only=True)
hc_processor = AutoImageProcessor.from_pretrained(hair_color_model_dir,local_files_only=True)
st_model = ViTForImageClassification.from_pretrained(skin_type_model_dir,local_files_only=True)
st_processor = AutoImageProcessor.from_pretrained(skin_type_model_dir,local_files_only=True)

class HairSkinClassifier(nn.Module):
    def __init__(self, disp,device):
        super().__init__()
        self.disp = disp
        self.kernel_sharp = np.array([[0, -1, 0],
                                      [-1, 5, -1],
                                      [0, -1, 0]])
        self.hc_processor = hc_processor
        self.st_processor = st_processor
        self.model_hair_color = hc_model
        self.model_skin_type = st_model

        self.shared_downlift_space = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=4, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((224, 224))
        )
        self.hair_color_head = nn.Linear(5,2) # Note: jasny / ciemny
        self.skin_color_head = nn.Linear(2,4) # Note: I,II,III i IV->VI

    def show_all_pic(self,images, disp):
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

    def ClaheHSV(self,image, clip, grid_size):
        hsv = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid_size)
        result = clahe.apply(v)
        hsv = cv2.merge((h, s, result))
        equalized_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return equalized_img

    def Enhance(self,images):
        img = []
        hist = []
        for i in range(len(images)):
            res = self.ClaheHSV(images[i], 5., (10, 10))
            norm = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX)
            img.append(norm)
        return img, hist

    def image_avg(self,images):
        avg_image = images[0]
        for i in range(len(images)):
            if i == 0:
                pass
            else:
                alpha = 1.0 / (i + 1)
                beta = 1.0 - alpha
                avg_image = cv2.addWeighted(images[i], alpha, avg_image, beta, 0.0)
        return avg_image

    def getnshow_dicom(self,ds, rgb, disp):
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
        self.show_all_pic(images, disp)
        return images

    def pre_processing(self,x):
        dsr = copy.deepcopy(x)
        dsg = copy.deepcopy(x)
        dsb = copy.deepcopy(x)
        imagesr = self.getnshow_dicom(dsr, 'r', self.disp)
        imagesg = self.getnshow_dicom(dsg, 'g', self.disp)
        imagesb = self.getnshow_dicom(dsb, 'b', self.disp)
        image_kernelr = []
        image_kernelg = []
        image_kernelb = []
        for i in range(len(imagesr)):
            image_kernelr.append(cv2.filter2D(src=imagesr[i], ddepth=-1, kernel=self.kernel_sharp))
            image_kernelg.append(cv2.filter2D(src=imagesg[i], ddepth=-1, kernel=self.kernel_sharp))
            image_kernelb.append(cv2.filter2D(src=imagesb[i], ddepth=-1, kernel=self.kernel_sharp))

        # Note ###################### # NORM
        images_normr, histr = self.Enhance(image_kernelr)
        images_normg, histg = self.Enhance(image_kernelg)
        images_normb, histb = self.Enhance(image_kernelb)

        # Note ###################### # 14xColor
        imgcol = np.zeros((14, 972, 1296, 3))
        imgcol[:, :, :, 0] = np.array(images_normr)[:, :, :, 0]
        imgcol[:, :, :, 1] = np.array(images_normg)[:, :, :, 1]
        imgcol[:, :, :, 2] = np.array(images_normb)[:, :, :, 2]

        #col_avg = self.image_avg(imgcol)
        avg_imager = self.image_avg(images_normr)
        avg_imageg = self.image_avg(images_normg)
        avg_imageb = self.image_avg(images_normb)

        norm_param = cv2.NORM_MINMAX
        avg_normb = cv2.normalize(avg_imageb, None, 0, 255, norm_param)
        avg_normg = cv2.normalize(avg_imageg, None, 0, 255, norm_param)
        avg_normr = cv2.normalize(avg_imager, None, 0, 255, norm_param)
        #avg_images = [avg_normb, avg_normg, avg_normr]

        image_comb = np.array(avg_imager)
        image_comb[:, :, 0] = avg_imager[:, :, 0]
        image_comb[:, :, 1] = avg_imageg[:, :, 1]
        image_comb[:, :, 2] = avg_imageb[:, :, 2]
        x = image_comb
        x_f32 = x.astype(np.float32) / 255.0
        return x_f32

    @staticmethod
    def witch_hair_color(x):
        result = torch.argmax(x,dim=1).item()
        if result == 0:
            hair_color = "DARK"
        else:
            hair_color = "LIGHT"
        return hair_color

    @staticmethod
    def witch_skin_color(x):
        result = torch.argmax(x, dim=1).item()
        if result == 0:
            skin_color = "TYPE I"
        elif result == 1:
            skin_color = "TYPE II"
        elif result == 2:
            skin_color = "TYPE III"
        else:
            skin_color = "TYPE IV+"
        return skin_color

    def forward(self, x,inference=True):
        x = self.pre_processing(x)
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        x = self.shared_downlift_space(x)
        x_hc = self.hc_processor(x, return_tensors="pt")
        x_hc = x_hc.pixel_values
        x_st = self.st_processor(x, return_tensors="pt")
        x_st = x_st.pixel_values

        x_hc = self.model_hair_color(x_hc).logits
        x_st = self.model_skin_type(x_st).logits

        x_hc = self.hair_color_head(x_hc)
        x_st = self.skin_color_head(x_st)

        if inference:
            hair_color = self.witch_hair_color(x_hc)
            skin_color = self.witch_skin_color(x_st)
        else:
            hair_color = "We do learning now, not inference!"
            skin_color = "We do learning now, not inference!"

        if self.disp == 1:
            plt.imshow(x)
            plt.show()
        else:pass
        return (x_hc, x_st),(hair_color, skin_color)
