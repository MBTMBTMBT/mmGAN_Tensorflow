import pickle
import numpy as np
import h5py
import data_loader
from settings import *


def create_hdf5_files(top_level_dir: str, output_dir: str, output_name=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if output_name is None:
        output_name = ORIGINAL_FILE_NAME

    # load train data
    LOGGER.info("loading train data")
    train_dir = os.path.join(top_level_dir, "training")
    sets = ['HGG', 'LGG']

    for each_set in sets:
        LOGGER.info(each_set)
        set_dir = os.path.join(train_dir, each_set)
        num_patients, patient_list, patient_dir_list = data_loader.detect_sample_dir(set_dir)
        # create the h5 file according to the number of the samples
        hdf5_file = h5py.File(os.path.join(output_dir, output_name + "_" + each_set + '.h5'), mode='w')
        grp = hdf5_file.create_group("original")
        grp.create_dataset("training_data",
                           (num_patients, 4, SPATIAL_SIZE_FOR_PREPROCESSING[0], SPATIAL_SIZE_FOR_PREPROCESSING[1], NUM_SLICES),
                           np.float32)
        grp.create_dataset("training_data_pat_name", (num_patients,), dtype='S100')
        grp.create_dataset("training_data_segmasks",
                           (num_patients, SPATIAL_SIZE_FOR_PREPROCESSING[0], SPATIAL_SIZE_FOR_PREPROCESSING[1], NUM_SLICES),
                           np.int16)
        hdf5_file_original = hdf5_file["original"]

        count_patients = 0
        for each_patient_dir in patient_dir_list:
            img, label, segmask = data_loader.load_patient(patient_list[count_patients], each_patient_dir,
                                                           SPATIAL_SIZE_FOR_PREPROCESSING, 4, NUM_SLICES)
            hdf5_file_original['training_data'][count_patients, ...] = img
            hdf5_file_original['training_data_segmasks'][count_patients, ...] = segmask
            hdf5_file_original['training_data_pat_name'][count_patients] = patient_list[count_patients].split('/')[
                -1].encode('utf-8')
            count_patients += 1

        # close the file to save memory
        # print(hdf5_file['original']['training_data'][0, 0])
        hdf5_file.close()

    # load validation data
    LOGGER.info("loading validation data")
    validation_dir = os.path.join(top_level_dir, "validation")
    num_patients, patient_list, patient_dir_list = data_loader.detect_sample_dir(validation_dir)
    # create the h5 file according to the number of the samples
    hdf5_file = h5py.File(os.path.join(output_dir, output_name + "_validation" + '.h5'), mode='w')
    grp = hdf5_file.create_group("original")
    grp.create_dataset("validation_data",
                       (num_patients, 4, SPATIAL_SIZE_FOR_PREPROCESSING[0], SPATIAL_SIZE_FOR_PREPROCESSING[1], NUM_SLICES),
                       np.float32)
    grp.create_dataset("validation_data_pat_name", (num_patients,), dtype='S100')
    hdf5_file_original = hdf5_file["original"]
    count_patients = 0

    for each_patient_dir in patient_dir_list:
        img, label, segmask = data_loader.load_patient(patient_list[count_patients], each_patient_dir,
                                                       SPATIAL_SIZE_FOR_PREPROCESSING, 4, NUM_SLICES)
        hdf5_file_original['validation_data'][count_patients, ...] = img
        hdf5_file_original['validation_data_pat_name'][count_patients] = patient_list[count_patients].split('/')[
            -1].encode(
            'utf-8')
        count_patients += 1

    # close the file to save memory
    hdf5_file.close()


def preprocess(read_dir: str, output_dir=None, read_name=None, output_name=None, mean_var_name=None):
    if output_dir is None:
        output_dir = read_dir
    if read_name is None:
        read_name = ORIGINAL_FILE_NAME
    if output_name is None:
        output_name = PREPROCESSED_FILE_NAME
    if mean_var_name is None:
        mean_var_name = MEAN_VAR_NAME

    LOGGER.info("start preprocessing")

    # preprocess training data (HGG and LGG)
    LOGGER.info("preprocess training data")
    sets = ["HGG", "LGG"]
    for each_set in sets:
        LOGGER.info("fetch %s dataset" % each_set)
        # create preprocessed file
        preprocessed_hdf5_file = h5py.File(os.path.join(read_dir, output_name + "_" + each_set + '.h5'), mode='w')
        preprocessed_grp = preprocessed_hdf5_file.create_group("preprocessed")

        # open original file
        original_hdf5_file = h5py.File(os.path.join(read_dir, read_name + "_" + each_set + '.h5'), mode='r')
        original_group = original_hdf5_file.require_group("original")

        # copy the names
        preprocessed_grp.create_dataset("training_data_pat_name", data=original_group["training_data_pat_name"])

        # create the new dataset for training data
        num_patients = original_group["training_data"].shape[0]
        preprocessed_grp.create_dataset("training_data", (num_patients, 4, SIZE_AFTER_CROPPING[0],
                                                          SIZE_AFTER_CROPPING[1], SIZE_AFTER_CROPPING[2]), np.float32)

        # crop the data
        LOGGER.info("cropping training data")
        coords = CROPPING_COORDINATES
        for i in range(0, num_patients):
            image = original_group["training_data"][i]
            cropped = image[:, coords[0]:coords[1], coords[2]:coords[3], coords[4]:coords[5]]
            preprocessed_grp["training_data"][i] = cropped
        LOGGER.info("dimension cropped from %s to %s"
                    % (original_group["training_data"].shape, preprocessed_grp["training_data"].shape))
        # LOGGER.info(original_group["training_data"].shape, end=" to ")
        # LOGGER.info(preprocessed_grp["training_data"].shape)
        # print(preprocessed_grp['training_data'][0, 0])

        # compute the mean and std value
        LOGGER.info("computing the mean and var values:")
        # image_np = np.empty(preprocessed_grp["training_data"].shape)
        # image_np[:] = preprocessed_grp["training_data"]
        _tmp, mean_var = standardize(preprocessed_grp["training_data"], find_mean_var_only=True,
                                     save_dump=os.path.join(output_dir, each_set + "_" + mean_var_name))
        LOGGER.info(mean_var)

        # create the new dataset for segmask
        num_patients = original_group["training_data_segmasks"].shape[0]
        preprocessed_grp.create_dataset("training_data_segmasks", (num_patients, SIZE_AFTER_CROPPING[0],
                                                                   SIZE_AFTER_CROPPING[1], SIZE_AFTER_CROPPING[2]),
                                        np.int16)

        # crop the segmask
        LOGGER.info("cropping segmask")
        coords = CROPPING_COORDINATES
        for i in range(0, num_patients):
            image = original_group["training_data_segmasks"][i]
            cropped = image[coords[0]:coords[1], coords[2]:coords[3], coords[4]:coords[5]]
            preprocessed_grp["training_data_segmasks"][i] = cropped
        LOGGER.info("dimension cropped from %s to %s"
                    % (original_group["training_data_segmasks"].shape, preprocessed_grp["training_data_segmasks"].shape))
        # LOGGER.info(original_group["training_data_segmasks"].shape, end=" to ")
        # LOGGER.info(preprocessed_grp["training_data_segmasks"].shape)

        # close the hdf5 files
        preprocessed_hdf5_file.close()
        original_hdf5_file.close()

    # preprocess validation data
    LOGGER.info("preprocess validation data")
    LOGGER.info("fetch validation dataset")

    # create preprocessed file
    preprocessed_hdf5_file = h5py.File(os.path.join(read_dir, output_name + "_" + "validation" + '.h5'), mode='w')
    preprocessed_grp = preprocessed_hdf5_file.create_group("preprocessed")

    # open original file
    original_hdf5_file = h5py.File(os.path.join(read_dir, read_name + "_" + "validation" + '.h5'), mode='r')
    original_group = original_hdf5_file.require_group("original")

    # create the new dataset for training data
    num_patients = original_group["validation_data"].shape[0]
    preprocessed_grp.create_dataset("validation_data", (num_patients, 4, SIZE_AFTER_CROPPING[0],
                                                        SIZE_AFTER_CROPPING[1], SIZE_AFTER_CROPPING[2]), np.float32)

    # copy the names
    preprocessed_grp.create_dataset("validation_data_pat_name", data=original_group["validation_data_pat_name"])

    # crop the data
    LOGGER.info("cropping validation data")
    coords = CROPPING_COORDINATES
    for i in range(0, num_patients):
        image = original_group["validation_data"][i]
        cropped = image[:, coords[0]:coords[1], coords[2]:coords[3], coords[4]:coords[5]]
        preprocessed_grp["validation_data"][i] = cropped
    LOGGER.info("dimension cropped from %s to %s"
                % (original_group["validation_data"].shape, preprocessed_grp["validation_data"].shape))
    # LOGGER.info(original_group["validation_data"].shape, end=" to ")
    # LOGGER.info(preprocessed_grp["validation_data"].shape)

    # close the hdf5 files
    preprocessed_hdf5_file.close()
    original_hdf5_file.close()


# copy from original data_loader.py
def standardize(images, find_mean_var_only=True, save_dump=None):
    """
    This function standardizes the input data to zero mean and unit variance. It is capable of calculating the
    mean and std values from the input data, or can also apply user specified mean/std values to the images.

    :param images: a h5py dataset with shape (num_qg, channels, x, y, z) to apply mean/std normalization to
    :param find_mean_var_only: only find the mean and variance of the input data, do not normalize
    :param save_dump: if True, saves the calculated mean/variance values to the disk in pickle form
    :return: standardized images, and vals (if mean/val was calculated by the function
    """

    # logger.info('Calculating mean value..')
    vals = {
        'mn': [],
        'var': []
    }

    # create np arrays to compute
    image_shape = images.shape
    sum_of_patients = np.zeros((image_shape[1], image_shape[2], image_shape[3], image_shape[4]), np.float32)
    mean = np.zeros(4, np.float32)
    patient_num = image_shape[0]

    # compute the sum
    for i in range(patient_num):
        sum_of_patients += images[i]

    # compute the mean
    for i in range(4):
        mean[i] = np.sum(sum_of_patients[i]) / (patient_num * image_shape[2] * image_shape[3] * image_shape[4])
        vals['mn'].append(mean[i])

    # logger.info('Calculating variance..')
    # compute the variance
    for i in range(4):
        temp_variance = np.zeros((image_shape[2], image_shape[3], image_shape[4]), np.float32)
        for j in range(patient_num):
            temp_variance += (images[j][i] - mean[i]) ** 2
        var = np.sum(temp_variance) / (patient_num * image_shape[2] * image_shape[3] * image_shape[4])
        vals['var'].append(var)

    # hgg - correct result (computed by the old method)
    # 'mn': [137.8238687678129, 139.06541255196476, 146.83794166571437, 81.9031688650977],
    # 'var': [91567.51291749951, 80731.20444536912, 112183.98736010231, 23409.12254743159]

    # not going to happen
    if not find_mean_var_only:
        # logger.info('Starting standardization process..')

        for i in range(4):
            images[:, i, :, :, :] = ((images[:, i, :, :, :] - vals['mn'][i]) / float(vals['var'][i]))

        # logger.info('Data standardized!')

    if save_dump is not None:
        # logger.info('Dumping mean and var values to disk..')
        pickle.dump(vals, open(save_dump, 'wb'))
    # logger.info('Done!')

    return images, vals


if __name__ == '__main__':
    output_path = os.path.join(TOP_LEVEL_PATH, "out")
    create_hdf5_files(TOP_LEVEL_PATH, output_path)
    preprocess(output_path)
