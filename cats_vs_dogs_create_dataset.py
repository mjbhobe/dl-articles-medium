#!usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cats_vs_dogs_create_dataset.py: code to organize images downloaded from Kaggle's Cats vs Dogs challenge
into a structure that Keras' ImageDataGenerator can use.

@author: Manish Bhobe
My experiments with Python, Machine Learning & Deep Learning.
This code is meant for education purposes only & is not intended for commercial/production use!
Use at your own risk!! I am not responsible if your CPU or GPU gets fried :D
"""
# imports and tweaks
import sys, os, random
import numpy as np

import kr_helper_funcs as kru

# seed random no generators
seed = 123
random.seed(seed)
np.random.seed(seed)

import warnings
warnings.filterwarnings('ignore')  # ignore all warnings

# globals for directory names
# this is the folder where all images from the Kaggle dataset were unzipped
# Change this to point to location where you unzipped Kaggle's train.zip file
IMAGES_ROOT = '/home/mjbhobe/code/python/pydata-book-master/learning-ml/data/kaggle/cat_or_dog/train'
assert os.path.exists(IMAGES_ROOT), "%s folder does not exist!" % images_root

# I will create the train/eval/test images under ./images subfolder of my pwd
MY_IMAGES_ROOT = './images/cats_vs_dogs'
if not os.path.exists(MY_IMAGES_ROOT): os.makedirs(MY_IMAGES_ROOT)
assert os.path.exists(MY_IMAGES_ROOT)

my_images_train_root = os.path.join(MY_IMAGES_ROOT,'train')
my_images_cat_train_root = os.path.join(my_images_train_root,'cat')
my_images_dog_train_root = os.path.join(my_images_train_root,'dog')

my_images_eval_root = os.path.join(MY_IMAGES_ROOT,'eval')
my_images_cat_eval_root = os.path.join(my_images_eval_root,'cat')
my_images_dog_eval_root = os.path.join(my_images_eval_root,'dog')

my_images_test_root = os.path.join(MY_IMAGES_ROOT,'test')
my_images_cat_test_root = os.path.join(my_images_test_root,'cat')
my_images_dog_test_root = os.path.join(my_images_test_root, 'dog')

# function to create datasets
def create_datasets(train_size=5000, eval_size=2000, test_size=500, clear_prev=True):
    """
    creates a reduced dataset of cat/dog images from the original Kaggle dataset
    @params:
        train_size: how many images to include in training images set (optional, default=5000)
        eval_size: how many images to include in cross-validation images set (optional, default=2000)
        test_size: how many images to include in test images set (optional, default=500)
        clear_prev: clear any previously created dataset? (optional, default:True)
    """
    def copy_images(image_set, source_folder, dest_folder, header_msg='Copying'):
        import shutil
        num_images = len(image_set)

        for i, image_name in enumerate(image_set):
            source_path = os.path.join(source_folder, image_name)
            dest_path = os.path.join(dest_folder, image_name)
            shutil.copyfile(source_path, dest_path)
            kru.progbar_msg(i, num_images, header_msg, image_name, final=False)
        kru.progbar_msg(i, num_images, header_msg, 'Done!', final=True)
        
    def make_my_folders(base_dir):
        for f1 in ['train', 'eval', 'test']:
            for f2 in ['cat', 'dog']:
                base_f1 = os.path.join(base_dir, f1)  # example base_dir/train, base_dir/cross_val etc
                if not os.path.exists(base_f1):
                    os.makedirs(base_f1)
                base_f1_f2 = os.path.join(base_f1, f2)     # example: base_dir/train/cat, base_dir/cross_val/dog
                if not os.path.exists(base_f1_f2):
                    os.makedirs(base_f1_f2)
        
    def clear_dir(directory):
        if os.path.exists(directory):
            print('  Clearing directory {}...'.format(directory))
            for the_file in os.listdir(directory):
                file_path = os.path.join(directory, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    else:
                        clear_dir(file_path)
                        os.rmdir(file_path)
                except Exception as e:
                    print(e)        
        
    image_names = np.array(os.listdir(IMAGES_ROOT))
    print('Found total of %d images in downloaded folder.' % len(image_names))
    
    assert len(image_names) > (train_size + eval_size + test_size)
    
    cat_image_names, dog_image_names = [], []
    for image_name in image_names:
        if image_name.lower().startswith('cat'):
            cat_image_names.append(image_name)
        elif image_name.lower().startswith('dog'):
            dog_image_names.append(image_name)
    print('Found %d cat images and %d dog images' % (len(cat_image_names), len(dog_image_names)))
    
    # shuffle the image names to get rid of any implicit sorting
    np.random.shuffle(cat_image_names)
    np.random.shuffle(dog_image_names)
    
    # now pick out train_size, eval_size & test_size from each image set
    cat_train_image_names = cat_image_names[:train_size]
    cat_eval_image_names = cat_image_names[train_size:(train_size+eval_size)]
    cat_test_image_names = cat_image_names[(train_size+eval_size):(train_size+eval_size+test_size)]
    
    assert len(cat_train_image_names) == train_size
    assert len(cat_eval_image_names) == eval_size
    assert len(cat_test_image_names) == test_size
    
    dog_train_image_names = dog_image_names[:train_size]
    dog_eval_image_names = dog_image_names[train_size:(train_size+eval_size)]
    dog_test_image_names = dog_image_names[(train_size+eval_size):(train_size+eval_size+test_size)]
    
    assert len(dog_train_image_names) == train_size
    assert len(dog_eval_image_names) == eval_size
    assert len(dog_test_image_names) == test_size
        
    if clear_prev:
        print('Deleting previous data sets...')
        clear_dir(MY_IMAGES_ROOT)
        if not os.path.exists(MY_IMAGES_ROOT): os.makedirs(MY_IMAGES_ROOT)
        make_my_folders(MY_IMAGES_ROOT)    
    
    # copy image sets
    print('Creating datasets...')
    copy_images(cat_train_image_names, IMAGES_ROOT, my_images_cat_train_root, 
                'Creating CAT training images in {} - '.format(my_images_cat_train_root))
    copy_images(dog_train_image_names, IMAGES_ROOT, my_images_dog_train_root, 
                'Creating DOG training images in {} - '.format(my_images_dog_train_root))
    
    copy_images(cat_eval_image_names, IMAGES_ROOT, my_images_cat_eval_root, 
                'Creating CAT eval images in {} - '.format(my_images_cat_eval_root))
    copy_images(dog_eval_image_names, IMAGES_ROOT, my_images_dog_eval_root, 
                'Creating DOG eval images in {} - '.format(my_images_dog_eval_root)) 

    copy_images(cat_test_image_names, IMAGES_ROOT, my_images_cat_test_root, 
                'Creating CAT test images in {} - '.format(my_images_cat_test_root))
    copy_images(dog_test_image_names, IMAGES_ROOT, my_images_dog_test_root, 
                'Creating DOG test images in {} - '.format(my_images_dog_test_root))    

# shows a sample of the source/downloaded images
image_names = np.array(os.listdir(IMAGES_ROOT))
indexes = np.random.permutation(range(len(image_names)))
image_names = image_names[indexes]
image_names[:25]

# create the datsets & delete any previously created datasets
create_datasets(clear_prev=True)