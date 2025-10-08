import numpy as np
import os
import matplotlib.pyplot as plt

from torchvision.datasets import CIFAR10
from torchvision import transforms

from skimage.color import rgb2gray
from skimage.feature import hog
from skimage import exposure

