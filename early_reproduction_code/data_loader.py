import os
import numpy as np
import SimpleITK as sitk
from settings import LOGGER


def detect_sample_dir(data_dir: str):  # data_dir e.g. "********//HGG"
    # detect the paths of each sample
    LOGGER.info("Reading under dir: %s" % data_dir)
    dir_list = os.listdir(data_dir)
    patient_list = []
    patient_dir_list = []

    for each_patient in dir_list:
        if 'Brats18' in each_patient:
            patient_list.append(each_patient)
            patient_dir_list.append(os.path.join(data_dir, each_patient))  # prepare the dir for each patient
            LOGGER.info("%s is discovered" % each_patient)

    num_of_patients = len(patient_list)
    LOGGER.info("%d patients discovered" % num_of_patients)
    return num_of_patients, patient_list, patient_dir_list


def load_patient(patient_name: str, patient_dir: str, out_shape: (), num_sequences: int, num_slices: int):
    LOGGER.info("loading patient %s" % patient_name)

    # find the contents of the patient
    image_dir_list = os.listdir(patient_dir)

    # create placeholders for images
    image = np.empty((num_sequences, out_shape[0], out_shape[1], num_slices)).astype(np.int16)
    label = np.empty(1).astype(np.int16)  # the label is kept here but it's actually not used
    seg_mask = np.empty((out_shape[0], out_shape[1], num_slices)).astype(np.int16)

    # load images of the patient
    # this part is referred to the original dataloader.py
    for each_image in image_dir_list:
        each_image_dir = os.path.join(patient_dir, each_image)

        # segmask is different from others so it is considered first
        if 'seg' in each_image:
            img_obj = sitk.ReadImage(each_image_dir)
            pix_data = sitk.GetArrayViewFromImage(img_obj)
            pix_data_swapped = np.swapaxes(pix_data, 0, 1)
            pix_data_swapped = np.swapaxes(pix_data_swapped, 1, 2)
            seg_mask[:, :, :] = pix_data_swapped

        else:  # consider the others
            if 't1.' in each_image:
                seq = 0
            elif 't2.' in each_image:
                seq = 1
            elif 't1ce.' in each_image:
                seq = 2
            elif 'flair.' in each_image:
                seq = 3
            else:
                LOGGER.info("invalid sequence: %s discovered at %s" % (each_image, patient_dir))
                continue
            img_obj = sitk.ReadImage(each_image_dir)
            # preprocessing function is in the original dataloader.py
            # img_obj = preprocessData(img_obj, process=preprocess)
            pix_data = sitk.GetArrayViewFromImage(img_obj)
            # print(pix_data)
            pix_data_swapped = np.swapaxes(pix_data, 0, 1)
            pix_data_swapped = np.swapaxes(pix_data_swapped, 1, 2)
            image[seq, :, :, :] = pix_data_swapped

    return image, label, seg_mask


def load_samples(data_dir: str, out_shape: (), num_sequences: int, num_slices: int):
    # num_slices might be the slices number of one brain?

    # load the paths of each sample
    LOGGER.info("Reading under dir: %s" % data_dir)
    dir_list = os.listdir(data_dir)
    patient_list = []
    for each_dir in dir_list:
        if 'Brats18' in each_dir:
            patient_list.append(each_dir)
            LOGGER.info("%s is discovered" % each_dir)
    num_of_patients = len(patient_list)
    LOGGER.info("%d patients discovered" % num_of_patients)

    # read images
    patient_count = 0
    LOGGER.info("loading images...")
    image_list = []
    label_list = []
    seg_mask_list = []
    for each_patient in patient_list:
        LOGGER.info("loading patient %d" % patient_count)
        patient_path = os.path.join(data_dir, each_patient)
        image_dir_list = os.listdir(patient_path)

        # create placeholders for images
        image = np.empty((1, num_sequences, out_shape[0], out_shape[1], num_slices)).astype(np.int16)
        label = np.empty(1, 1).astype(np.int16)
        seg_mask = np.empty((1, out_shape[0], out_shape[1], num_slices)).astype(np.int16)

        # load images of each patient
        # this part is referred to the original dataloader.py
        for each_image in image_dir_list:
            each_image_dir = os.path.join(patient_path, each_image)

            # segmask is different from others so it is considered first
            if 'seg' in each_image:
                img_obj = sitk.ReadImage(each_image_dir)
                pix_data = sitk.GetArrayViewFromImage(img_obj)
                pix_data_swapped = np.swapaxes(pix_data, 0, 1)
                pix_data_swapped = np.swapaxes(pix_data_swapped, 1, 2)
                seg_mask[:, :, :] = pix_data_swapped
                seg_mask_list.append(seg_mask)

            else:  # consider the others
                if 't1.' in each_image:
                    seq = 0
                elif 't2.' in each_image:
                    seq = 1
                elif 't1ce.' in each_image:
                    seq = 2
                elif 'flair.' in each_image:
                    seq = 3
                else:
                    LOGGER.info("invalid sequence: %s discovered at %s" % (each_image, patient_path))
                    continue
                img_obj = sitk.ReadImage(each_image_dir)
                # preprocessing function is in the original dataloader.py
                # img_obj = preprocessData(img_obj, process=preprocess)
                pix_data = sitk.GetArrayViewFromImage(img_obj)
                pix_data_swapped = np.swapaxes(pix_data, 0, 1)
                pix_data_swapped = np.swapaxes(pix_data_swapped, 1, 2)
                image[seq, :, :, :] = pix_data_swapped
                image_list.append(image)
                label_list.append(label)

        # go for the next patient
        patient_count += 1

        return num_of_patients, image_list, label_list, seg_mask_list


if __name__ == '__main__':
    load_samples(r"C:\Users\13769\Desktop\PROGRAMS\python\new_mmgan\BRATS2018\Training\HGG", (240, 240), num_sequences=4, num_slices=155)
