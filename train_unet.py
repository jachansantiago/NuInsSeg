#import libs
import os
from glob import glob
import numpy as np
from sklearn.model_selection import KFold
import time  
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.layers import *
from random import shuffle 
# import matplotlib.pyplot as plt
import pandas as pd
from data import DataGenerator
from eval_utils import compute_metrics, step_decay_schedule
from unet import deep_unet

## disabeling warning msg
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
import warnings
warnings.simplefilter('ignore')
# import sys
# sys.stdout.flush() # resolving tqdm problem
import argparse
from pprint import pprint

import json

def main(opts):

    base_path = 'nuinsseg/'
    organ_names = [ name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name)) ]

    # input and outpu paths
    img_path = glob('{}*{}'.format('nuinsseg/*/tissue images/', 'png'))
    binary_mask_path = glob('{}*{}'.format('nuinsseg/*/mask binary/', 'png'))
    distance_mask_path = glob('{}*{}'.format('nuinsseg/*/distance maps/', 'png'))
    label_mask_path = glob('{}*{}'.format('nuinsseg/*/label masks modify/', 'tif'))
    vague_mask_path =  glob('{}*{}'.format('nuinsseg/*/vague areas/mask binary/', 'png'))


    img_path.sort()
    binary_mask_path.sort()
    distance_mask_path.sort()
    label_mask_path.sort()
    vague_mask_path.sort()


    # create folders to save the best models and images (if needed) for each fold
    if not os.path.exists(opts['model_save_path']):
        os.makedirs(opts['model_save_path'])
    if not os.path.exists(opts['result_save_path']):
        os.makedirs(opts['result_save_path'])    
    if not os.path.exists(opts['result_save_path']+ 'validation/unet'):
        os.makedirs(opts['result_save_path'] + 'validation/unet')
    if not os.path.exists(opts['result_save_path']+ 'validation/watershed_unet'):
        os.makedirs(opts['result_save_path'] + 'validation/watershed_unet')

    #random check
    rand_num = np.random.randint(len(img_path))
    print('image path: {}\n'.format(img_path[rand_num]),
        'binary mask path: {}\n'.format(binary_mask_path[rand_num]),
        'distance mask path: {}\n'.format(distance_mask_path[rand_num]),
        'label mask path: {}\n'.format(label_mask_path[rand_num]))
    
    # main training loop (for all k fold cross-validation)
    kf = KFold(n_splits= opts['k_fold'],random_state= opts['random_seed_num'],shuffle=True)
    kf.get_n_splits(img_path)

    start_time = time.time()
    current_fold = 1

    fold_metrics = []

    for idx, [train_index,  test_index] in enumerate(kf.split(img_path)):
        shuffle(train_index)
        shuffle(test_index)

        train_img   = [img_path[name] for name in train_index]
        train_mask  = [binary_mask_path[name] for name in train_index]
        train_dis   = [distance_mask_path[name] for name in train_index]
        train_label = [label_mask_path[name] for name in train_index]
        
        test_img   = [img_path[name] for name in test_index]
        test_mask  = [binary_mask_path[name] for name in test_index]
        test_dis   = [distance_mask_path[name] for name in test_index]
        test_label = [label_mask_path[name] for name in test_index]
        test_vague = [vague_mask_path[name] for name in test_index]
        
        #creating validation set
        validation_set_img = []
        validation_set_label = []
        #validation_DIS = []
        validation_set_vague = []
        for counter in range(len(test_img)):
            val_img = cv2.imread(test_img[counter])
            val_img = cv2.cvtColor(val_img, cv2.COLOR_BGR2RGB)
            val_img = val_img/255
            val_label = cv2.imread(test_label[counter], -1) # cv2.IMREAD_UNCHANGED: 
            #It specifies to load an image as such including alpha channel. 
            #Alternatively, we can pass integer value -1 for this flag.
            val_vague = cv2.imread(test_vague[counter], -1)
            
            validation_set_img.append(val_img)
            validation_set_label.append(val_label)
            validation_set_vague.append(val_vague)
            
        validation_set_img = np.array(validation_set_img)
        validation_set_label = np.array(validation_set_label)
        validation_set_vague = np.array(validation_set_vague)
        
        model_path = opts['model_save_path'] + 'unet_{}.weights.h5'.format(current_fold)
        logger = CSVLogger(opts['model_save_path']+ 'unet_{}.log'.format(current_fold))
        LR_drop = step_decay_schedule(initial_lr= opts['init_LR'], 
                                decay_factor = opts['LR_decay_factor'], 
                                epochs_drop = opts['LR_drop_after_nth_epoch'])
        model_raw = deep_unet(opts['number_of_channel'], opts['init_LR'])
        checkpoint = ModelCheckpoint(model_path, monitor='val_dice_coef', verbose=1,
                                save_best_only=True, mode='max', save_weights_only = True)

        train_gen =  DataGenerator(train_img,
                    train_mask,
                    1,
                    1,
                    opts['crop_size'], opts['crop_size'],
                    distance_unet_flag=0,
                    augment=True,
                    BACKBONE_model= '',
                    use_pretrain_flag= False)
        validation_gen = DataGenerator(test_img,
                    test_mask,
                    1,
                    1,
                    opts['crop_size'], opts['crop_size'],
                    distance_unet_flag=0,
                    augment= False,
                    BACKBONE_model= '',
                    use_pretrain_flag= False)

        dataset_size = len(train_img)
        steps_per_epoch = dataset_size // opts['batch_size']

        # get generator output signatures
        example = next(train_gen())
        example = tf.convert_to_tensor(example[0])
        example_mask = next(train_gen())
        example_mask = tf.convert_to_tensor(example_mask[1])
        output_signature = (tf.TensorSpec(shape=(example.shape), dtype=tf.float32),
                            tf.TensorSpec(shape=(example_mask.shape), dtype=tf.float32))

        train_dataset = tf.data.Dataset.from_generator(train_gen, output_signature=output_signature)
        train_dataset = train_dataset.batch(opts['batch_size'])

        validation_dataset = tf.data.Dataset.from_generator(validation_gen, output_signature=output_signature)
        validation_dataset = validation_dataset.batch(opts['batch_size'])
        history = model_raw.fit(train_dataset,
                                    validation_data=validation_dataset,
                                    validation_steps=1,
                                    epochs=opts['epoch_num'], verbose=1,
                                    callbacks=[checkpoint, logger, LR_drop], steps_per_epoch=steps_per_epoch)
        
        model_raw.load_weights(opts['model_save_path'] + 'unet_{}.weights.h5'.format(current_fold))

        ## predication on validation set
        pred_val = model_raw.predict(validation_set_img, verbose=1, batch_size=1)

        opts['result_save_path_current_fold'] = os.path.join(opts['result_save_path'], 'fold_{}'.format(current_fold))
        if not os.path.exists(opts['result_save_path_current_fold']):
            os.makedirs(opts['result_save_path_current_fold'])

        metrics = compute_metrics(pred_val, validation_set_label, validation_set_vague, threshold=opts['threshold'],  opts=opts)
        pprint(metrics)
        with open(os.path.join(opts['result_save_path'], 'validation', f'metrics_fold_{current_fold}.json'), 'w') as f:
            json.dump(metrics, f)
        fold_metrics.append(metrics)
        current_fold = current_fold + 1

    df = pd.DataFrame(fold_metrics)
    df.to_csv('unet_metrics.csv', index=False)

    finish_time = time.time() 
    print('==========') 
    print('total training time (all 5 folds): {:.2f} minutes'.format((finish_time- start_time)/60))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train U-Net model for image segmentation')
    parser.add_argument('--number_of_channel', type=int, default=3, help='Number of input channels')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for segmentation')
    parser.add_argument('--epoch_num', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--quick_run', type=int, default=1, help='Quick run flag (0 or 1)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--random_seed_num', type=int, default=19, help='Random seed number for reproducibility')
    parser.add_argument('--k_fold', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--save_val_results', type=int, default=1, help='Flag to save validation results (0 or 1)')
    parser.add_argument('--init_LR', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--LR_decay_factor', type=float, default=0.5, help='Learning rate decay factor')
    parser.add_argument('--LR_drop_after_nth_epoch', type=int, default=20, help='Epoch after which to drop learning rate')
    parser.add_argument('--crop_size', type=int, default=512, help='Crop size for input images')
    parser.add_argument('--result_save_path', type=str, default='unet_prediction_image/', help='Path to save prediction images')
    parser.add_argument('--model_save_path', type=str, default='unet_output_model/', help='Path to save model weights')
    opts = vars(parser.parse_args())
    main(opts)
                    
