# encoding=utf-8
"""
imageai_prediction.py: predicting the class of an image with ImageAI library

@author: Manish Bhobe
My experiments with Python, Data Science, Machine Learning and Deep Learning
"""
from imageai.Prediction import ImagePrediction
import matplotlib.pyplot as plt
from PIL import Image
import random
import numpy as np
import tensorflow as tf
import os
import sys
import warnings
import time
warnings.filterwarnings('ignore')

seed = 123
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

# replace this path with full path of folder where you downloaded pretrained weights
# from https://github.com/OlafenwaMoses/ImageAI/releases/tag/1.0/
IMAGEAI_MODELS_HOME = "C:\\Dev\\Code\\Python\\pydata-book-master\\learning-ml\\dl-tensorflow\\pretrained_models"
IMAGEAI_MODEL = 'resnet50_weights_tf_dim_ordering_tf_kernels.h5'

assert os.path.exists(
    IMAGEAI_MODELS_HOME), "%s - pretrained models dir does not exist!" % IMAGEAI_MODELS_HOME

MODEL_PATH = os.path.join(IMAGEAI_MODELS_HOME, IMAGEAI_MODEL)
assert os.path.exists(MODEL_PATH), "%s - unable find pre-trained model weights here!" % MODEL_PATH

image_file_names = [
    "car-1.jpg",
    "cat-5.jpg",
    "bird-3.jpg",
    "dog-2.jpg",
    "bird-2.jpg"
]

# instantiate the predictor & load weights
# we are using ResNet50
predictor = ImagePrediction()
predictor.setModelTypeAsResNet()
predictor.setModelPath(MODEL_PATH)
predictor.loadModel()


def display_predictions(predictor, image_path, pred_count=10, fig_size=(8, 6)):
    assert os.path.exists(image_path)
    assert fig_size is not None
    assert pred_count > 0

    img = Image.open(image_path)
    plt.figure(figsize=fig_size)
    plt.imshow(img)
    plt.title(image_path)

    # display predictions
    predictions, probabilities = predictor.predictImage(image_path, result_count=pred_count)
    print('Predictions: %s' % image_path)
    for pred, prob in zip(predictions, probabilities):
        print('  - %s: prob: %.3f' % (pred, prob))


for image_file_name in image_file_names:
    test_image = os.path.join(os.getcwd(), image_file_name)
    display_predictions(predictor, test_image)
