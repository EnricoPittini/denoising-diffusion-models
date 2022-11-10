import glob
import os
import shutil

from distutils.dir_util import copy_tree
from tqdm.auto import tqdm
from time import sleep 

from torchvision.datasets import ImageFolder


def load_imagenet(root: str, split: str):
    os.system('wget http://cs231n.stanford.edu/tiny-imagenet-200.zip')
    os.system('unzip -qq tiny-imagenet-200.zip')

    os.system(f'mv tiny-imagenet-200 /{root}')

    path_dataset = os.path.join(root, 'tiny-imagenet-200',split)

    if split=='train':
        _arrange_train_images(path_dataset)
    elif split=='val':
        _split_validation_in_folders(path_dataset)

    os.remove('tiny-imagenet-200.zip')

    return ImageFolder(path_dataset)
        


def _arrange_train_images(path_data: str) -> int:
    """Moves images from the folder images to the category folder.

    Args:
        path_data: path to the training set.

    Returns:
        the number of moved images.

    Raises:
        FileNotFoundError: is path_data is empty.
    """
    num_moved_images = 0
    try:
        directories = os.listdir(path_data)
    except FileNotFoundError:
        print(f'No directories found in {path_data}')
        raise
        
    for i in tqdm(range(len(directories)), leave=False):        
        path_category = os.path.join(path_data, directories[i])
        path_old_images = os.path.join(path_category, 'images')
        paths = glob.glob(os.path.join(path_old_images, '*'))
        
        for j in tqdm(range(len(paths)), leave=False):
            path_source = paths[j]
            name_file = os.path.basename(path_source)
            path_dest = os.path.join(path_category, name_file)
            shutil.move(path_source, path_dest)            
            num_moved_images += 1

        if os.path.exists(path_old_images):
            os.rmdir(path_old_images)
    return num_moved_images

def _split_validation_in_folders(path_data: str) -> int:
    """Splits the validation data set in category folders.

    Args:
        path_data: path to the validation set.

    Returns:
        the number of moved images.

    Raises:
        FileNotFoundError: if val_annotations.txt is not found.
        FileNotFoundError: if validation images are not found.
    """
    num_moved_images = 0
    image_to_label = {}
    path_file_annotation = os.path.join(path_data, 'val_annotations.txt')
    with open(path_file_annotation, 'r') as f:
        for line in f.readlines():
            split_line = line.split('\t')
            image_to_label[split_line[0]] = split_line[1]

    path_val = os.path.join(path_data, 'images')
    paths = glob.glob(os.path.join(path_val, '*'))
    if not paths:
        raise FileNotFoundError(f'No validation images found in {path_val}')

    for path in paths:
        file = path.split(os.sep)[-1]
        folder = image_to_label[file]
        path_folder = os.path.join(path_data, str(folder))
        if not os.path.exists(path_folder):
            os.mkdir(path_folder)

    for i in tqdm(range(len(paths)), leave=False):
        path_source = paths[i]
        file = path_source.split(os.sep)[-1]
        folder = image_to_label[file]
        path_dest = os.path.join(path_data, str(folder), str(file))
        shutil.move(path_source, path_dest)        
        num_moved_images += 1

    os.remove(path_file_annotation)
    os.rmdir(path_val)
    return num_moved_images