import os
import multiprocessing

from mm_gan.constants_and_tools import NII_SHAPE, CHANNELS, TXT_OUT_NAMES
import numpy as np
import logging
import random

from io_helpers import load_nii


class DataFetcher(object):

    def __init__(self, group_txt_dir: str, group_txt_names=None, channels=None, batch_size=8,
                 patients_in_buffer=16, nii_shape=None, slice_cut_off=(12, 12), img_shape=(256, 256), name="fetcher",
                 parallel=True, shuffle=True):

        print("fetcher created...")
        self.name = name
        self.shuffle = shuffle

        # get txt file paths
        if group_txt_names is None:
            group_txt_names = TXT_OUT_NAMES
        self.txt_paths = multiprocessing.Manager().list()
        self.group_names = multiprocessing.Manager().list()
        for each_txt_name in group_txt_names:
            self.txt_paths.append(os.path.join(group_txt_dir, each_txt_name))
            self.group_names.append(each_txt_name.split('.')[0])

        if channels is None:
            channels = CHANNELS
        self.channels = channels

        self.batch_size = batch_size
        self.patients_in_buffer = patients_in_buffer

        # get range of slice
        if nii_shape is None:
            self.nii_shape = NII_SHAPE
        else:
            self.nii_shape = nii_shape
        final_slice_num = self.nii_shape[0] - slice_cut_off[0] - slice_cut_off[1]
        self.slice_cut_off = slice_cut_off
        self.load_img_shape = (final_slice_num, len(self.channels), self.nii_shape[1], self.nii_shape[2])
        self.img_shape = img_shape

        # output info
        self.info()

        # prepare lists of patients' dirs
        self.patient_dirs = multiprocessing.Manager().list()
        self.patient_dirs_lock = multiprocessing.Manager().Lock()
        self.load_patient_dir_list()
        if self.shuffle:
            random.shuffle(self.patient_dirs)

        # full patient buffer
        self.patients_buffer = multiprocessing.Manager().list()
        self.patients_dirs_in_buffer = multiprocessing.Manager().list()
        self.patients_buffer_and_dir_buffer_lock = multiprocessing.Manager().Lock()

        # buffer of batches
        self.batches_buffer = multiprocessing.Manager().list()
        self.batches_buffer_lock = multiprocessing.Manager().Lock()

        # batches ready to give
        self.batches_ready = multiprocessing.Manager().list()

        # initialize the buffer
        self.load()
        self.batches_ready = self.batches_buffer
        self.batches_buffer = multiprocessing.Manager().list()

        # the loader process
        self.parallel = parallel
        self.loader_process = None

        if self.parallel:
            self.loader_process = multiprocessing.Process(target=self.load())
            self.loader_process.start()
        else:
            self.load()

    def reset(self):
        print("fetcher reset...")
        self.info()
        if self.loader_process is not None:
            self.loader_process.join()
            # self.loader_process.close()

        # prepare lists of patients' dirs
        self.patient_dirs = multiprocessing.Manager().list()
        self.patient_dirs_lock = multiprocessing.Manager().Lock()
        self.load_patient_dir_list()
        if self.shuffle:
            random.shuffle(self.patient_dirs)

        # full patient buffer
        self.patients_buffer = multiprocessing.Manager().list()
        self.patients_dirs_in_buffer = multiprocessing.Manager().list()
        self.patients_buffer_and_dir_buffer_lock = multiprocessing.Manager().Lock()

        # buffer of batches
        self.batches_buffer = multiprocessing.Manager().list()
        self.batches_buffer_lock = multiprocessing.Manager().Lock()

        # batches ready to give
        self.batches_ready = multiprocessing.Manager().list()

        # initialize the buffer
        self.load()
        self.batches_ready = self.batches_buffer
        self.batches_buffer = multiprocessing.Manager().list()

        if self.parallel:
            self.loader_process = multiprocessing.Process(target=self.load())
            self.loader_process.start()
        else:
            self.load()

    def info(self):
        print("fetcher - %s" % self.name)
        print("batch size: %d; patients' num in buffer: %d;" % (self.batch_size, self.patients_in_buffer))
        s = self.nii_shape + self.load_img_shape
        print("nii shape: (%d, %d, %d); load img shape: (%d, %d, %d, %d)" % (s[0], s[1], s[2], s[3], s[4], s[5], s[6]))

    def load_patient_dir_list(self):
        for idx, each_txt_path in enumerate(self.txt_paths):
            # self.logger.debug("loading patients' path from %s" % self.group_names[idx])
            file = open(each_txt_path, 'r')
            next_path = file.readline()
            while next_path:
                next_path = next_path.replace('\n', '')
                self.patient_dirs_lock.acquire()
                self.patient_dirs.append(next_path)
                self.patient_dirs_lock.release()
                next_path = file.readline()
            file.close()

    def load_patients_to_buffer(self):
        for i in range(self.patients_in_buffer):
            # self.patient_dirs_lock.acquire()
            patients_left = len(self.patient_dirs)
            # self.patient_dirs_lock.release()
            if patients_left == 0:
                return
            channel_dirs = []
            # self.patient_dirs_lock.acquire()
            patient_dir = self.patient_dirs.pop(0)
            # self.patient_dirs_lock.release()
            for c in self.channels:
                channel_dirs.append(os.path.join(patient_dir, "%s_standardized.nii.gz" % c))

            # self.patients_buffer_and_dir_buffer_lock.acquire()
            self.patients_buffer.append(load_full_patient(channel_dirs, self.nii_shape))
            self.patients_dirs_in_buffer.append(patient_dir)
            # self.patients_buffer_and_dir_buffer_lock.release()
            # self.logger.debug("%s - fetch patient: " % self.name, patient_dir)

    def load_batches_to_buffer(self):
        batch_shape = (self.batch_size, len(self.channels), self.nii_shape[1], self.nii_shape[2])
        buffer_slice_num = self.patients_in_buffer * self.load_img_shape[0]
        buffer_shape = (buffer_slice_num, len(self.channels), self.nii_shape[1], self.nii_shape[2])
        buffer_arr = np.empty(buffer_shape, dtype='float32')
        for idx in range(self.patients_in_buffer):
            self.patients_buffer_and_dir_buffer_lock.acquire()
            if len(self.patients_buffer) == 0:
                self.patients_buffer_and_dir_buffer_lock.release()
                return
            each_patient_arr = self.patients_buffer.pop(0)
            each_patient_dir = self.patients_dirs_in_buffer.pop(0)
            self.patients_buffer_and_dir_buffer_lock.release()
            # print("pop: ", each_patient_dir)
            buffer_arr[idx * self.load_img_shape[0]: idx * self.load_img_shape[0] + self.load_img_shape[0]] \
                = each_patient_arr[self.slice_cut_off[0]: self.slice_cut_off[0] + self.load_img_shape[0]]
        if self.shuffle:
            np.random.shuffle(buffer_arr)
        # print(buffer_arr.shape[0] // batch_shape[0])
        self.batches_buffer_lock.acquire()
        for i in range(buffer_arr.shape[0] // batch_shape[0]):
            batch = buffer_arr[i * batch_shape[0]: i * batch_shape[0] + batch_shape[0]]
            self.batches_buffer.append(batch)
            # print("write")
            # print("buffer: ", len(self.batches_buffer))
        self.batches_buffer_lock.release()

    def load(self):
        if len(self.patients_buffer) == 0:
            self.patients_buffer_and_dir_buffer_lock.acquire()
            self.patients_buffer = multiprocessing.Manager().list()
            self.patients_dirs_in_buffer = multiprocessing.Manager().list()
            self.patients_buffer_and_dir_buffer_lock.release()
            self.load_patients_to_buffer()
        self.load_batches_to_buffer()
        return

    def get_next_batch(self) -> np.ndarray:
        if len(self.batches_ready) == 0 and self.parallel and self.loader_process is not None:
            self.loader_process.join()
            # self.loader_process.close()
        if len(self.batches_ready) != 0:
            next_batch = self.batches_ready.pop(0)
            batch_temp = np.ones(shape=(next_batch.shape[0], next_batch.shape[1], self.img_shape[0], self.img_shape[1]),
                                 dtype='float32') * next_batch[0, 0, 0, 0]
            size_diff = (self.img_shape[0] - next_batch.shape[2], self.img_shape[1] - next_batch.shape[3])
            if size_diff[0] > 0 and size_diff[1] > 0:
                down = (size_diff[0] // 2, size_diff[1] // 2)
                up = (next_batch.shape[2] + size_diff[0] // 2, next_batch.shape[3] + size_diff[1] // 2)
                # print(down, up)
                batch_temp[:, :, down[0]:up[0], down[1]:up[1]] = next_batch
            else:
                down = (-size_diff[0] // 2, -size_diff[1] // 2)
                up = (self.img_shape[0] - size_diff[0] // 2, self.img_shape[1] - size_diff[1] // 2)
                batch_temp = next_batch[:, :, down[0]:up[0], down[1]:up[1]]
            next_batch = batch_temp
            if len(self.batches_ready) == 0:
                if len(self.batches_buffer) != 0:
                    self.batches_buffer_lock.acquire()
                    self.batches_ready = self.batches_buffer
                    self.batches_buffer = multiprocessing.Manager().list()
                    self.batches_buffer_lock.release()
                    if self.parallel:
                        self.loader_process = multiprocessing.Process(target=self.load)
                        self.loader_process.start()
                    else:
                        self.load()
            return next_batch
        else:
            return None


def load_full_patient(nii_paths: list, nii_shape: tuple) -> np.ndarray:
    channels = len(nii_paths)
    shape = (channels, nii_shape[0], nii_shape[1], nii_shape[2])
    full_patient = np.empty(shape=shape, dtype='float32')
    for i in range(channels):
        channel_img = load_nii(nii_paths[i])
        full_patient[i] = channel_img
    # move the slice dimension to the first
    full_patient = np.transpose(full_patient, axes=(1, 0, 2, 3))
    return full_patient


if __name__ == '__main__':
    # get logger
    test_logger = logging.getLogger(name="test_logger")
    test_logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()  # stderr output to console
    formatter = logging.Formatter('[pid:%(process)d]-%(asctime)s: %(message)s', "%y%b%d-%H:%M:%S")
    stream_handler.setFormatter(formatter)
    test_logger.addHandler(stream_handler)
    test_logger.info("hey")
    fetcher = DataFetcher(r'D:\my_files\programmes\python\BRATS2018_normalized', name="test")
                          # group_txt_names=['group0_standardized.txt'])

    count = 0
    while True:
        test_next_batch = fetcher.get_next_batch()
        if test_next_batch is None:
            break
        else:
            count += 1
            print(count, test_next_batch.shape)
