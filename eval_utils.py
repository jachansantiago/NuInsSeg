#import libs
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
from data import get_id_from_file_path
from skimage.color import label2rgb

# define loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
#####################################################################################
# Combination of Dice and binary cross entophy loss function that is used in this baseline segmentation
def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


# learning rate scheduler
def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, epochs_drop=1000):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/epochs_drop))
    
    return LearningRateScheduler(schedule, verbose = 1)


# evaluation index (from: https://github.com/vqdang/hover_net/blob/master/src/metrics/stats_utils.py)
#############################################################################################################

def get_fast_aji(true, pred):
    """AJI version distributed by MoNuSeg, has no permutation problem but suffered from 
    over-penalisation similar to DICE2.
    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4] 
    not [2, 3, 6, 10]. Please call `remap_label` before hand and `by_size` flag has no 
    effect on the result.
    """
    true = np.copy(true)  # ? do we need this
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))
    #print(len(pred_id_list))
    if len(pred_id_list) == 1:
        return 0

    true_masks = [None,]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [None,]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_inter = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    pairwise_union = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            pairwise_inter[true_id - 1, pred_id - 1] = inter
            pairwise_union[true_id - 1, pred_id - 1] = total - inter

    pairwise_iou = pairwise_inter / (pairwise_union + 1.0e-6)
    # pair of pred that give highest iou for each true, dont care
    # about reusing pred instance multiple times
    paired_pred = np.argmax(pairwise_iou, axis=1)
    pairwise_iou = np.max(pairwise_iou, axis=1)
    # exlude those dont have intersection
    paired_true = np.nonzero(pairwise_iou > 0.0)[0]
    paired_pred = paired_pred[paired_true]
    # print(paired_true.shape, paired_pred.shape)
    overall_inter = (pairwise_inter[paired_true, paired_pred]).sum()
    overall_union = (pairwise_union[paired_true, paired_pred]).sum()

    paired_true = list(paired_true + 1)  # index to instance ID
    paired_pred = list(paired_pred + 1)
    # add all unpaired GT and Prediction into the union
    unpaired_true = np.array(
        [idx for idx in true_id_list[1:] if idx not in paired_true]
    )
    unpaired_pred = np.array(
        [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    )
    for true_id in unpaired_true:
        overall_union += true_masks[true_id].sum()
    for pred_id in unpaired_pred:
        overall_union += pred_masks[pred_id].sum()

    aji_score = overall_inter / overall_union
    #print(aji_score)
    return aji_score

#############################################################################################################
def get_fast_pq(true, pred, match_iou=0.5):
    """`match_iou` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique 
    (1 prediction instance to 1 GT instance mapping).
    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing. 
    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.    
    
    Fast computation requires instance IDs are in contiguous orderding 
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand 
    and `by_size` flag has no effect on the result.
    Returns:
        [dq, sq, pq]: measurement statistic
        [paired_true, paired_pred, unpaired_true, unpaired_pred]: 
                      pairing information to perform measurement
                    
    """
    assert match_iou >= 0.0, "Cant' be negative"

    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))
    
    if len(pred_id_list) == 1:
        return [0, 0, 0], [0,0, 0, 0]

    true_masks = [
        None,
    ]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [
        None,
    ]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)

    # prefill with value
    pairwise_iou = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return [dq, sq, dq * sq], [paired_true, paired_pred, unpaired_true, unpaired_pred]


#############################################################################################################
def get_dice_1(true, pred):
    """Traditional dice."""
    # cast to binary 1st
    true = np.copy(true)
    pred = np.copy(pred)
    true[true > 0] = 1
    pred[pred > 0] = 1
    inter = true * pred
    denom = true + pred
    dice_score = 2.0 * np.sum(inter) / (np.sum(denom) + 0.0001)
    if np.sum(inter)==0 and np.sum(denom)==0:
        dice_score = 1 # to handel cases without any nuclei
    #print(dice_score)
    return dice_score
#############################################################################################################
def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.
    Args:
        pred    : the 2d array contain instances where each instances is marked
                  by non-zero integer
        by_size : renaming with larger nuclei has smaller id (on-top)
    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


#############################################################################################################
def pair_coordinates(setA, setB, radius):
    """Use the Munkres or Kuhn-Munkres algorithm to find the most optimal 
    unique pairing (largest possible match) when pairing points in set B 
    against points in set A, using distance as cost function.
    Args:
        setA, setB: np.array (float32) of size Nx2 contains the of XY coordinate
                    of N different points 
        radius: valid area around a point in setA to consider 
                a given coordinate in setB a candidate for match
    Return:
        pairing: pairing is an array of indices
        where point at index pairing[0] in set A paired with point
        in set B at index pairing[1]
        unparedA, unpairedB: remaining poitn in set A and set B unpaired
    """
    # * Euclidean distance as the cost matrix
    pair_distance = scipy.spatial.distance.cdist(setA, setB, metric='euclidean')

    # * Munkres pairing with scipy library
    # the algorithm return (row indices, matched column indices)
    # if there is multiple same cost in a row, index of first occurence 
    # is return, thus the unique pairing is ensured
    indicesA, paired_indicesB = linear_sum_assignment(pair_distance)

    # extract the paired cost and remove instances 
    # outside of designated radius
    pair_cost = pair_distance[indicesA, paired_indicesB]

    pairedA = indicesA[pair_cost <= radius]
    pairedB = paired_indicesB[pair_cost <= radius]

    pairing = np.concatenate([pairedA[:,None], pairedB[:,None]], axis=-1)
    unpairedA = np.delete(np.arange(setA.shape[0]), pairedA)
    unpairedB = np.delete(np.arange(setB.shape[0]), pairedB)
    return pairing, unpairedA, unpairedB

def compute_metrics(pred_val, validation_set_label, validation_set_vague, threshold=0.5, opts=None):
    pred_val_t = (pred_val >  threshold).astype(np.uint8)
    output_watershed_tot_fold = []
    output_watershed_tot_fold_wo_vague = []
    validation_set_label_tot_fold_wo_vague = []
    dice_scores = []
    aji_scores = []
    pq_scores = []

    dice_watershed_scores = []
    aji_watershed_scores = []
    pq_watershed_scores = []

    dice_watershed_wo_vague_scores = []
    aji_watershed_wo_vague_scores = []
    pq_watershed_wo_vague_scores = []
    for val_len in tqdm.tqdm(range(len(pred_val))):
        # with watershed post processing
        # local_maxi = peak_local_max(np.squeeze(pred_val[val_len]),
                                    # exclude_border=False, footprint=np.ones((15, 15)))
        local_maxi = peak_local_max(np.squeeze(pred_val[val_len]),
                                    exclude_border=False, footprint=np.ones((15, 15)))
        peaks_mask = np.zeros_like(np.squeeze(pred_val[val_len]), dtype=bool)
        peaks_mask[local_maxi] = True

        marker = ndi.label(peaks_mask)[0]
        output_watershed = watershed(-np.squeeze(pred_val[val_len]), marker,mask = np.squeeze(pred_val_t[[val_len]]))
        output_watershed[np.squeeze(pred_val_t[[val_len]])==0] = 0
        output_watershed = remove_small_objects(output_watershed, min_size=50, connectivity=2)#remove small objects
        
        

        # without post processing 
        output_raw_0 = np.squeeze(pred_val_t[val_len])
        output_raw = skimage.morphology.label(output_raw_0)
        output_raw = remove_small_objects(output_raw, min_size=50, connectivity=2) #remove small objects


        output_watershed = remap_label(output_watershed)
        validation_set_label[val_len] = remap_label(validation_set_label[val_len])
        output_raw = remap_label(output_raw)

        dice_scores.append(get_dice_1(validation_set_label[val_len], output_raw))
        aji_scores.append(get_fast_aji(validation_set_label[val_len], output_raw))
        pq_scores.append(get_fast_pq(validation_set_label[val_len], output_raw)[0][2])
        

        dice_watershed_scores.append(get_dice_1(validation_set_label[val_len],output_watershed))
        aji_watershed_scores.append(get_fast_aji(validation_set_label[val_len], output_watershed))
        pq_watershed_scores.append(get_fast_pq(validation_set_label[val_len], output_watershed)[0][2])


        output_watershed_wo_vague = np.copy(output_watershed)
        output_watershed_wo_vague[validation_set_vague[val_len] == 255] = 0
        output_watershed_wo_vague = remove_small_objects(output_watershed_wo_vague, min_size=50, connectivity=2) 
        
        validation_set_label_wo_vague = np.copy(validation_set_label[val_len])
        validation_set_label_wo_vague[validation_set_vague[val_len] == 255] = 0
        validation_set_label_wo_vague = remove_small_objects(validation_set_label_wo_vague, min_size=50, connectivity=2)

        
        
        output_watershed_wo_vague = remap_label(output_watershed_wo_vague)
        validation_set_label_wo_vague = remap_label(validation_set_label_wo_vague)
        
        # test_name = get_id_from_file_path(test_img[val_len],'.png' )
        # print('output_raw shape:', output_raw.shape)
        # print('output_raw stats:', output_raw.min(), output_raw.max(), output_raw.min(axis=0), output_raw.max(axis=0)


        
        
        ###################################################################
        
        
        dice_watershed_wo_vague_scores.append(get_dice_1(validation_set_label_wo_vague,output_watershed_wo_vague))
        aji_watershed_wo_vague_scores.append(get_fast_aji(validation_set_label_wo_vague, output_watershed_wo_vague))    
        pq_watershed_wo_vague_scores.append(get_fast_pq(validation_set_label_wo_vague, output_watershed_wo_vague)[0][2])

        output_watershed_tot_fold.append(output_watershed)
        output_watershed_tot_fold_wo_vague.append(output_watershed_wo_vague)
        validation_set_label_tot_fold_wo_vague.append(validation_set_label_wo_vague)

                
        output_raw_rgb = label2rgb(output_raw, bg_label=0)
        output_raw_rgb = (output_raw_rgb * 255).astype(np.uint16)
        output_watershed_rgb = label2rgb(output_watershed, bg_label=0)
        output_watershed_rgb = (output_watershed_rgb * 255).astype(np.uint16)

        gt_rgb = label2rgb(validation_set_label[val_len], bg_label=0)
        gt_rgb = (gt_rgb * 255).astype(np.uint16)
        # save in label image
        # print('output_raw_rgb shape:', output_raw_rgb.shape)
        imsave(opts['result_save_path_current_fold']+'/watershed_sam_{}.png'.format(val_len), output_watershed_rgb.astype(np.uint8))
        imsave(opts['result_save_path_current_fold']+'/sam_{}.png'.format(val_len), output_raw_rgb.astype(np.uint8))
        imsave(opts['result_save_path_current_fold']+'/gt_{}.png'.format(val_len), gt_rgb.astype(np.uint8))

    metrics = {}
    metrics['dice_scores'] = np.mean(dice_scores)
    metrics['aji_scores'] = np.mean(aji_scores)
    metrics['pq_scores'] = np.mean(pq_scores)

    metrics['dice_watershed_scores'] = np.mean(dice_watershed_scores)
    metrics['aji_watershed_scores'] = np.mean(aji_watershed_scores)
    metrics['pq_watershed_scores'] = np.mean(pq_watershed_scores)

    metrics['dice_watershed_wo_vague_scores'] = np.mean(dice_watershed_wo_vague_scores)
    metrics['aji_watershed_wo_vague_scores'] = np.mean(aji_watershed_wo_vague_scores)
    metrics['pq_watershed_wo_vague_scores'] = np.mean(pq_watershed_wo_vague_scores)

    return metrics
