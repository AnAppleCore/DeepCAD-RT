from deepcad.train_collection import training_class
from deepcad.movie_display import display
from deepcad.utils import get_first_filename,download_demo

datasets_path='/data/yanhongwei/SIM/noisy/train'

n_epochs = 10               # the number of training epochs
GPU = '1'                   # the index of GPU used for computation (e.g. '0', '0,1', '0,1,2')
train_datasets_size = 6000  # dataset size for training (the number of patches)
patch_xy = 150              # the width and height of 3D patches
patch_t = 150               # the time dimension of 3D patches
overlap_factor = 0.25       # the overlap factor between two adjacent patches
pth_dir = '/data/yanhongwei/SSR_results'           # pth file and visualization result file path
num_workers = 4             # if you use Windows system, set this to 0.

visualize_images_per_epoch = False  # choose whether to show inference performance after each epoch
save_test_images_per_epoch = False  # choose whether to save inference image after each epoch in pth path

# playing the first noise movie using opencv.
display_images = True

train_dict = {
    # dataset dependent parameters
    'patch_x': patch_xy,
    'patch_y': patch_xy,
    'patch_t': patch_t,
    'overlap_factor':overlap_factor,
    'scale_factor': 1,                  # the factor for image intensity scaling
    'select_img_num': 100000,           # select the number of images used for training (use all frames by default)
    'train_datasets_size': train_datasets_size,
    'datasets_path': datasets_path,
    'pth_dir': pth_dir,
    # network related parameters
    'n_epochs': n_epochs,
    'lr': 0.00005,                       # initial learning rate
    'b1': 0.5,                           # Adam: bata1
    'b2': 0.999,                         # Adam: bata2
    'fmap': 16,                          # the number of feature maps
    'GPU': GPU,
    'num_workers': num_workers,
    'visualize_images_per_epoch': visualize_images_per_epoch,
    'save_test_images_per_epoch': save_test_images_per_epoch,
    'ss_stride': 10, 
    'mask_type': 'rectangle',
}

# %%% Training preparation
# first we create a training class object with the specified parameters
tc = training_class(train_dict)
# start the training process
tc.run()