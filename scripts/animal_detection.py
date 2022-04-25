#!/usr/bin/env python

'''
File: animal_detection.py
Author: Kevin Lessard, Abdiel Fernandez, Rose Guay Hottin, Santino Nanini

Data cleaning, model creation, model training and model evaluation module for animal detection using Faster R-CNN.

'''

# Imports
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torchmetrics.detection.map import MeanAveragePrecision
from PIL import Image
import pycocotools

# Imports local modules downloaded from TorchVision repo v0.8.2, references/detection
from lib import utils
from lib import transforms
from lib import coco_utils
from lib import coco_eval
from lib.engine import train_one_epoch, evaluate

# Set the paths
output_path = '../output'
img_folder = '../eccv_18_all_images_sm'

cis_test_ann_path = '../eccv_18_annotation_files/cis_test_annotations.json'
cis_val_ann_path = '../eccv_18_annotation_files/cis_val_annotations.json'
train_ann_path = '../eccv_18_annotation_files/train_annotations.json'
trans_test_ann_path = '../eccv_18_annotation_files/trans_test_annotations.json'
trans_val_ann_path = '../eccv_18_annotation_files/trans_val_annotations.json'

# Load the data
cis_test_ann = json.load(open(cis_test_ann_path))
cis_val_ann = json.load(open(cis_val_ann_path))
train_ann= json.load(open(train_ann_path))
trans_test_ann = json.load(open(trans_test_ann_path))
trans_val_ann = json.load(open(trans_val_ann_path))

# test that everything is imported temporarily
print(len(cis_test_ann['images']))
print(len(cis_val_ann['images']))
print(len(train_ann['images']))
print(len(trans_test_ann['images']))
print(len(trans_val_ann['images']))


# Model class
class model:
    pass
