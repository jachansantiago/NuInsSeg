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
import torchvision
import json

# pretty print
from pprint import pprint

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
from torch.utils.data import DataLoader, Dataset
import torch
from sam_finetune import train_model, get_sam_model

from tqdm import tqdm
try:
    # Disable all GPUs
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except RuntimeError as e:
    # Handle the exception if GPUs are already initialized
    print(e)

class TFDatasetAsTorch(Dataset):
    def __init__(self, tf_dataset):
        self.data = list(tf_dataset.as_numpy_iterator())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, mask = self.data[idx]
        image = torch.from_numpy(image).permute(2, 0, 1)  # Change to (C, H, W)
        mask = torch.from_numpy(mask).permute(2, 0, 1) # Change to (C, H, W)
        # Convert to float32
        image = image.float()
        mask = mask.long()
        # print('image shape: ', image.shape, image.dtype, image.min(), image.max())
        # print('mask shape: ', mask.shape, mask.dtype, mask.min(), mask.max())
        return {"image": image, "mask": mask}
    

def evaluate_model(model, val_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i,data in enumerate(tqdm(val_loader)):
            imgs = data['image'].cuda()
            msks = torchvision.transforms.Resize((256,256))(data['mask'])
            msks = msks.cuda()

            img_emb= model.image_encoder(imgs)
            sparse_emb, dense_emb = model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
            pred, _ = model.mask_decoder(
                            image_embeddings=img_emb,
                            image_pe=model.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=sparse_emb,
                            dense_prompt_embeddings=dense_emb, 
                            multimask_output=True,
                            )
            pred = pred.argmax(dim=1).cpu()
            predictions.append(pred)
    predictions = torch.cat(predictions, dim=0).unsqueeze(dim=-1).permute(0, 3, 1, 2) # B, 256, 256, 1
    # resize predictions to 512x512
    predictions = torch.nn.functional.interpolate(predictions.float(), size=(512, 512), mode='nearest').permute(0, 2, 3, 1) # B, 512, 512, 1
    print('predictions shape: ', predictions.shape, predictions.dtype, predictions.min(), predictions.max())
    return predictions.numpy()


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
    if not os.path.exists(opts['result_save_path']+ 'validation/sam'):
        os.makedirs(opts['result_save_path'] + 'validation/sam')
    if not os.path.exists(opts['result_save_path']+ 'validation/watershed_sam'):
        os.makedirs(opts['result_save_path'] + 'validation/watershed_sam')

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
        print('validation set image shape: ', validation_set_img.shape, validation_set_img.dtype, validation_set_img.min(), validation_set_img.max())
        validation_set_label = np.array(validation_set_label)
        validation_set_vague = np.array(validation_set_vague)
        
        # model_path = opts['model_save_path'] + 'unet_{}.weights.h5'.format(current_fold)
        # logger = CSVLogger(opts['model_save_path']+ 'unet_{}.log'.format(current_fold))
        # LR_drop = step_decay_schedule(initial_lr= opts['init_LR'], 
        #                         decay_factor = opts['LR_decay_factor'], 
        #                         epochs_drop = opts['LR_drop_after_nth_epoch'])
        # model_raw = deep_unet(opts['number_of_channel'], opts['init_LR'])
        # checkpoint = ModelCheckpoint(model_path, monitor='val_dice_coef', verbose=1,
                                # save_best_only=True, mode='max', save_weights_only = True)

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
                    augment=False,
                    BACKBONE_model= '',
                    use_pretrain_flag= False)

        dataset_size = len(train_img)
        steps_per_epoch = dataset_size // opts['batch_size']

        # get generator output signatures
        example = next(train_gen())
        example = tf.convert_to_tensor(example[0])
        example_mask = next(train_gen())
        example_mask = tf.convert_to_tensor(example_mask[1])
        print('example shape: ', example.shape)
        print('example mask shape: ', example_mask.shape)
        output_signature = (tf.TensorSpec(shape=(example.shape), dtype=tf.float32),
                            tf.TensorSpec(shape=(example_mask.shape), dtype=tf.float32))

        train_dataset = tf.data.Dataset.from_generator(train_gen, output_signature=output_signature)
        train_loader = DataLoader(TFDatasetAsTorch(train_dataset), batch_size=opts['batch_size'], shuffle=True)

        validation_dataset = tf.data.Dataset.from_generator(validation_gen, output_signature=output_signature)
        val_loader = DataLoader(TFDatasetAsTorch(validation_dataset), batch_size=opts['batch_size'], shuffle=False)

        for batch in train_loader:
            print('batch shape: ', batch['image'].shape, batch['mask'].shape, batch['image'].dtype, batch['mask'].dtype)
            print('batch image stats: ', batch['image'].min(), batch['image'].max(), batch['image'].mean(), batch['image'].std())
            print('batch mask stats: ', batch['mask'].min(), batch['mask'].max())

            break

        for batch in val_loader:
            print('val batch shape: ', batch['image'].shape, batch['mask'].shape)
            print('val batch shape: ', batch['image'].dtype, batch['mask'].dtype, batch['image'].min(), batch['image'].max(), batch['image'].mean(), batch['image'].std())
            print('val batch shape: ', batch['image'].shape, batch['mask'].shape, batch['image'].dtype, batch['mask'].dtype)
            print('val batch image stats: ', batch['image'].min(), batch['image'].max(), batch['image'].mean(), batch['image'].std())
            print('val batch mask stats: ', batch['mask'].min(), batch['mask'].max())
            break
        model_folder = os.path.join(opts['model_save_path'], 'sam_{}'.format(current_fold))
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        model = get_sam_model()

        if opts['eval_only'] == True:
            # load the model
            model_weights = os.path.join(model_folder, 'checkpoint_best.pth')
            model.load_state_dict(torch.load(model_weights, map_location='cuda:0'))
            model = model.cuda()
            model.eval()
        else:
            model = train_model(model, train_loader, val_loader,  model_folder ,opts['epoch_num'])
        
        predictions = evaluate_model(model, val_loader)

        # batch resize to 256x256
        # validation_set_label = np.array([cv2.resize(validation_set_label[i], (256, 256), interpolation=cv2.INTER_NEAREST) for i in range(len(validation_set_label))])
        # print('validation set label shape: ', validation_set_label.shape, validation_set_label.dtype, validation_set_label.min(), validation_set_label.max())
        # set vague in 256
        # validation_set_vague = np.array([cv2.resize(validation_set_vague[i], (256, 256), interpolation=cv2.INTER_NEAREST) for i in range(len(validation_set_vague))])
        # print('validation set vague shape: ', validation_set_vague.shape, validation_set_vague.dtype, validation_set_vague.min(), validation_set_vague.max())
        # upscale prediction
        # print('predictions shape: ', predictions[0].shape, predictions[0].dtype, predictions.min(), predictions[0].max())
        opts['result_save_path_current_fold'] = os.path.join(opts['result_save_path'], 'fold_{}'.format(current_fold))
        if not os.path.exists(opts['result_save_path_current_fold']):
            os.makedirs(opts['result_save_path_current_fold'])
        metrics = compute_metrics(predictions, validation_set_label, validation_set_vague, threshold=opts['threshold'], opts=opts)
        pprint(metrics)
        # save metrics to csv
        # df = pd.DataFrame(metrics)
        # df.to_csv(opts['result_save_path'] + 'validation/metrics_fold_{}.csv'.format(current_fold), index=False)
        # save as json
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
    parser.add_argument('--epoch_num', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--quick_run', type=int, default=1, help='Quick run flag (0 or 1)')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training')
    parser.add_argument('--random_seed_num', type=int, default=19, help='Random seed number for reproducibility')
    parser.add_argument('--k_fold', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--save_val_results', type=int, default=1, help='Flag to save validation results (0 or 1)')
    parser.add_argument('--init_LR', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--LR_decay_factor', type=float, default=0.5, help='Learning rate decay factor')
    parser.add_argument('--LR_drop_after_nth_epoch', type=int, default=20, help='Epoch after which to drop learning rate')
    parser.add_argument('--crop_size', type=int, default=1024, help='Crop size for input images')
    parser.add_argument('--result_save_path', type=str, default='sam_prediction_image/', help='Path to save prediction images')
    parser.add_argument('--model_save_path', type=str, default='sam_output_model/', help='Path to save model weights')
    parser.add_argument('--eval_only', default=False, action="store_true", help='Flag to evaluate only (True or False)')
    opts = vars(parser.parse_args())
    main(opts)
                    
