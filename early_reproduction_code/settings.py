import os
import logging
import time

# default top-level path
# TOP_LEVEL_PATH = os.getcwd() + r'\BRATS2018'
TOP_LEVEL_PATH = '/home/shen/Dataset/mri/BRATS2018'
DATASET_PATH = os.path.join(TOP_LEVEL_PATH, 'out')

# default names for the paths
ORIGINAL_FILE_NAME = 'BRATS2018_original'
PREPROCESSED_FILE_NAME = 'BRATS2018_Cropped_Normalized_preprocessed'
MEAN_VAR_NAME = 'mean_var.p'

# training size and number of slices
SPATIAL_SIZE_FOR_PREPROCESSING = (240, 240)
SPATIAL_SIZE_FOR_TRAINING = (256, 256)
NUM_SLICES = 155
RESIZE_SLICES = 148

# cropping coordinates
CROPPING_COORDINATES = [29, 223, 41, 196, 0, 148]
SIZE_AFTER_CROPPING = [194, 155, 148]

# training parameters
ARCH = 'model_pycharm_test'
N_CPUS = 0
PIN_MEMORY = True
TRAIN_PATIENTS_HGG = 200
TRAIN_PATIENTS_LGG = 70
TEST_PATIENTS_HGG = 10
TEST_PATIENTS_LGG = 5
ADAM_LEARNING_RATE = 0.0002
ADAM_BETA_1 = 0.5
ADAM_BETA_2 = 0.999
BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
START_EPOCH = 0
RUN_EPOCHS = 60
CURRICULUM_LEARNING = True
IMPUTATION = "zeros"
IMPLICIT_CONDITIONING = True
LAMBDA = 0.9  # variable that sets the relative importance to loss_GAN and loss_pixel

# others
# SAVE_PARAMETERS_INTERVAL = 3
SAVE_PARAMETERS_EPOCH = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
SAVE_IMG = True
SAVE_IMG_SLICE = 48
LOG_LEVEL = 'info'
LOG_TO_File = True

# make output dir
if not os.path.isdir(DATASET_PATH):
    os.mkdir(DATASET_PATH)

# get logger
try:
    LOGGER = logging.getLogger(__file__.split('/')[-1])
except:
    LOGGER = logging.getLogger(__name__)

if 'info' in LOG_LEVEL:
    logging.basicConfig(level=logging.INFO)
elif 'debug' in LOG_LEVEL:
    logging.basicConfig(level=logging.DEBUG)

if LOG_TO_File:
    log_name = "log_%s.txt" % str(time.strftime("%y_%b_%d_%H_%M_%S"))
    if not os.path.isdir(os.path.join(DATASET_PATH, 'logs')):
        os.mkdir(os.path.join(DATASET_PATH, 'logs'))
    handler = logging.FileHandler(os.path.join(DATASET_PATH, 'logs', log_name))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)


if __name__ == '__main__':
    print("Top-level path: " + TOP_LEVEL_PATH)
