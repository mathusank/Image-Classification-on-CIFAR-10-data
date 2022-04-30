# CIFAR-10 Dataset with Xception, Resnet and Densenet

Resnet and Densenet architecture will be loaded from the PyTorch models library. Xception architecture will be made from scratch.

All that is needed to run this code is a Notebook such as Jupyter, Google Collab, etc.

## Installation

Use the package manager [pip]
!pip install einops > /dev/null

## Dependencies
import pickle<br />
import numpy as np
from tqdm.notebook import tqdm
from PIL import Image
import cv2
import scipy.ndimage as nd
import pandas as pd
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import einops
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import torch.nn.functional as fonctional

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

## Authors

Mathusan Kathirithamby , 
Ali Akbar Sabzi Dizajyekan ,
Majdi Saghrouni ,
Oussama Soukeuyr
