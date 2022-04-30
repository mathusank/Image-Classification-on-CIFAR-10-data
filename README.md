# CIFAR-10 Dataset with Xception, Resnet and Densenet

Resnet and Densenet architecture will be loaded from the PyTorch models library. Xception architecture will be made from scratch.

All that is needed to run this code is a Notebook such as Jupyter, Google Collab, etc.

## Installation

Use the package manager [pip] <br />
!pip install einops > /dev/null

## Dependencies
import pickle<br />
import numpy as np <br />
from tqdm.notebook import tqdm <br />
from PIL import Image <br />
import cv2 <br />
import scipy.ndimage as nd <br />
import pandas as pd <br />
from skimage.feature import local_binary_pattern <br />
import matplotlib.pyplot as plt <br />
import einops <br />
import os <br />
<br /><br />
from sklearn.model_selection import train_test_split <br />
from sklearn.model_selection import GridSearchCV <br />
from sklearn.model_selection import RandomizedSearchCV <br />
from sklearn.cluster import KMeans <br />
from sklearn.svm import SVC <br />
from sklearn import preprocessing <br />
from sklearn.preprocessing import Normalizer <br />
from sklearn.ensemble import RandomForestClassifier <br />
from sklearn.decomposition import IncrementalPCA <br />
from sklearn.metrics import accuracy_score <br />
from sklearn.metrics import f1_score <br />

import torch <br />
import torchvision <br />
import torchvision.transforms as transforms <br />
from torch.utils.data import DataLoader, random_split <br />
from torchvision import transforms <br />
from torch.utils.data import Dataset, DataLoader <br />
import torch.nn as nn <br />
<br />
import torch.nn.functional as fonctional <br />
<br />
import torch.optim as optim <br />
from torch.optim.lr_scheduler import ReduceLROnPlateau <br />

## Authors

[Mathusan Kathirithamby](https://github.com/mathusank), <br />
Ali Akbar Sabzi Dizajyekan ,<br />
Majdi Saghrouni ,<br />
Oussama Soukeuyr
