import pandas as pd
import os

TRAIN_PATH = './dataset/train'
TRAIN_IMG_AMOUNT = 685

VAL_PATH = './dataset/val'
VAL_IMG_AMOUNT = 170

PATH = './dataset/Raw Data'
IMG_AMOUNT = 855

EXT = '.png'

train_dataset = {'low_res': [f'{TRAIN_PATH}/low_res/{i}{EXT}' for i in range(TRAIN_IMG_AMOUNT)],
                 'high_res': [f'{TRAIN_PATH}/high_res/{i}{EXT}' for i in range(TRAIN_IMG_AMOUNT)]}

val_dataset = {'low_res': [f'{VAL_PATH}/low_res/{i}{EXT}' for i in range(VAL_IMG_AMOUNT)],
               'high_res': [f'{VAL_PATH}/high_res/{i}{EXT}' for i in range(VAL_IMG_AMOUNT)]}

dataset = {'low_res': [f'{PATH}/low_res/{i}{EXT}' for i in range(IMG_AMOUNT)],
           'high_res': [f'{PATH}/high_res/{i}{EXT}' for i in range(IMG_AMOUNT)]}

train_df = pd.DataFrame(train_dataset)
val_df = pd.DataFrame(val_dataset)
df = pd.DataFrame(dataset)

train_df.to_csv('train.csv', index=False)
val_df.to_csv('val.csv', index=False)
df.to_csv('dataset.csv', index=False)
