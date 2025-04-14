import os
from glob import glob
import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold
import time  
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from albumentations import *
from tensorflow.keras import backend as K
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.segmentation import watershed
import skimage.morphology
from skimage.io import imsave
from skimage.morphology import remove_small_objects
import tqdm
from random import shuffle 
import matplotlib.pyplot as plt
import pandas as pd


# augmentation function
def albumentation_aug(p=1.0, crop_size_row = 448, crop_size_col = 448 ):
    return Compose([
        Resize(crop_size_row, crop_size_col, always_apply=True, p=1),
        RandomCrop(crop_size_row, crop_size_col, always_apply=True, p=1),
        CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
        RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, brightness_by_max=True, p=0.4),
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.1),
        HorizontalFlip(always_apply=False, p=0.5),
        VerticalFlip(always_apply=False, p=0.5),
        RandomRotate90(p=0.5),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, interpolation=1, 
                         border_mode=4, always_apply=False, p=0.1),

    ], p=p)

def albumentation_aug_eval(p=1.0, crop_size_row = 448, crop_size_col = 448 ):
    return Compose([
        Resize(crop_size_row, crop_size_col, always_apply=True, p=1),
        # RandomCrop(crop_size_row, crop_size_col, always_apply=True, p=1),
        # CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
        # RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, brightness_by_max=True, p=0.4),
        # HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=0.1),
        # HorizontalFlip(always_apply=False, p=0.5),
        # VerticalFlip(always_apply=False, p=0.5),
        # RandomRotate90(p=0.5),
        # ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, interpolation=1, 
        #                  border_mode=4, always_apply=False, p=0.1),

    ], p=p)


# data generator related functions
def get_id_from_file_path(file_path, indicator):
    return file_path.split(os.path.sep)[-1].replace(indicator, '')
#############################################################################################################
def chunker(seq, seq2, size):
    return ([seq[pos:pos + size], seq2[pos:pos + size]] for pos in range(0, len(seq), size))
#############################################################################################################
def data_gen(list_files, list_files2, batch_size, p , size_row, size_col, distance_unet_flag = 0,
             augment= False, BACKBONE_model = None, use_pretrain_flag = 1):
    crop_size_row = size_row
    crop_size_col = size_col
    aug = albumentation_aug(p, crop_size_row, crop_size_col)

    # while True:
    for batch in chunker(list_files,list_files2, batch_size):
        X = []
        Y = []

        for count in range(len(batch[0])):
            x = cv2.imread(batch[0][count])
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x_mask = cv2.imread(batch[1][count], cv2.IMREAD_GRAYSCALE)
            
            x_mask_temp = np.zeros((x_mask.shape[0], x_mask.shape[1]))
            x_mask_temp[x_mask == 255] = 1
            

            if distance_unet_flag == False:
                if augment:
                    augmented = aug(image= x, mask= x_mask_temp)
                    x = augmented['image']
                    if use_pretrain_flag == 1:
                        x = preprocess_input(x)
                    x_mask_temp = augmented['mask']
                    x = x/255
                else:
                    x = x/255    
                X.append(x)
                Y.append(x_mask_temp)
            else:
                if augment:
                    augmented = aug(image=x, mask=x_mask)
                    x = augmented['image']
                    if use_pretrain_flag == 1:
                        x = preprocess_input(x)
                    x_mask = augmented['mask']
                    x = x/255
                else:
                    x = x/255  
                    
                X.append(x)
                x_mask = (x_mask - np.min(x_mask))/ (np.max(x_mask) - np.min(x_mask) + 0.0000001)
                Y.append(x_mask)

            del x_mask
            del x_mask_temp
            del x
        Y = np.expand_dims(np.array(Y), axis=3)
        Y = np.array(Y)
        yield np.array(X).astype(np.float32), np.array(Y).astype(np.float32)

class DataGenerator:
    def __init__(self, list_files, list_files2, batch_size, p, size_row, size_col, 
                 distance_unet_flag=0, augment=False, BACKBONE_model=None, use_pretrain_flag=1):
        self.list_files = list_files
        self.list_files2 = list_files2
        self.batch_size = batch_size
        self.p = p
        self.size_row = size_row
        self.size_col = size_col
        self.distance_unet_flag = distance_unet_flag
        self.augment = augment
        self.BACKBONE_model = BACKBONE_model
        self.use_pretrain_flag = use_pretrain_flag
        self.aug = albumentation_aug(p, size_row, size_col)
        self.aug_eval = albumentation_aug_eval(p, size_row, size_col)

    def __call__(self):
        # while True:
        for batch in chunker(self.list_files, self.list_files2, self.batch_size):
            X = []
            Y = []

            for count in range(len(batch[0])):
                x = cv2.imread(batch[0][count])
                x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                x_mask = cv2.imread(batch[1][count], cv2.IMREAD_GRAYSCALE)

                x_mask_temp = np.zeros((x_mask.shape[0], x_mask.shape[1]))
                x_mask_temp[x_mask == 255] = 1

                if not self.distance_unet_flag:
                    if self.augment:
                        augmented = self.aug(image=x, mask=x_mask_temp)
                        x = augmented['image']
                        if self.use_pretrain_flag:
                            x = preprocess_input(x)
                        x_mask_temp = augmented['mask']
                        # print("x_mask_temp", x_mask_temp.shape, x_mask_temp.dtype, x_mask_temp.min(), x_mask_temp.max())
                        x = x / 255
                    else:
                        augmented = self.aug_eval(image=x, mask=x_mask_temp)
                        x = augmented['image']
                        if self.use_pretrain_flag:
                            x = preprocess_input(x)
                        x_mask_temp = augmented['mask']
                        # print("x_mask_temp", x_mask_temp.shape, x_mask_temp.dtype, x_mask_temp.min(), x_mask_temp.max())
                        x = x / 255
                    X.append(x)
                    Y.append(x_mask_temp)
                else:
                    if self.augment:
                        augmented = self.aug(image=x, mask=x_mask)
                        x = augmented['image']
                        if self.use_pretrain_flag:
                            x = preprocess_input(x)
                        x_mask = augmented['mask']
                        # print("x_mask", x_mask.shape, x_mask.dtype, x_mask.min(), x_mask.max())
                        x = x / 255
                    else:
                        augmented = self.aug_eval(image=x, mask=x_mask)
                        x = augmented['image']
                        if self.use_pretrain_flag:
                            x = preprocess_input(x)
                        x_mask = augmented['mask']
                        # print("x_mask", x_mask.shape, x_mask.dtype, x_mask.min(), x_mask.max())
                        x = x / 255
                    X.append(x)
                    x_mask = (x_mask - np.min(x_mask)) / (np.max(x_mask) - np.min(x_mask) + 1e-7)
                    Y.append(x_mask)

                del x_mask
                del x_mask_temp
                del x

            Y = np.expand_dims(np.array(Y), axis=3)
            yield np.array(X).astype(np.float32)[0], np.array(Y).astype(np.float32)[0]