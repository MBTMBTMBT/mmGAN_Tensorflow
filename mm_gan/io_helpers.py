import SimpleITK as sitk
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
try:
    import mm_gan.constants_and_tools as constants_and_tools
except ModuleNotFoundError:
    import constants_and_tools as constants_and_tools


def load_nii(nii_path: str) -> np.ndarray:
    img = sitk.ReadImage(nii_path)
    img_arr = sitk.GetArrayFromImage(img)
    return img_arr


def write_nii(img_arr: np.ndarray, nii_path: str):
    img = sitk.GetImageFromArray(img_arr)
    sitk.WriteImage(img, nii_path)


def decode_tfrecord(example, batch_size):
    # get feature dict
    feature_description = {
        "name": tf.io.FixedLenFeature((), tf.string),
        "t1": tf.io.FixedLenFeature((), tf.string),
        "t2": tf.io.FixedLenFeature((), tf.string),
        "t1ce": tf.io.FixedLenFeature((), tf.string),
        "flair": tf.io.FixedLenFeature((), tf.string),
    }
    feature_dict = tf.io.parse_example(example, feature_description)

    # resolve values
    name = feature_dict["name"]
    t1 = tf.reshape(tf.io.decode_raw(
        feature_dict["t1"], out_type=tf.float32), shape=(batch_size, 1, 256, 256))
    t2 = tf.reshape(tf.io.decode_raw(
        feature_dict["t2"], out_type=tf.float32), shape=(batch_size, 1, 256, 256))
    t1ce = tf.reshape(tf.io.decode_raw(
        feature_dict["t1ce"], out_type=tf.float32), shape=(batch_size, 1, 256, 256))
    flair = tf.reshape(tf.io.decode_raw(
        feature_dict["flair"], out_type=tf.float32), shape=(batch_size, 1, 256, 256))
    return name, t1, t2, t1ce, flair


def get_dataset(tfrecords: list, compression_type: str, batch_size: int, drop_remainder: bool,
                shuffle: bool, buffer_size: int) -> tf.data.TFRecordDataset:
    dataset = None
    for each_record in tfrecords:
        new_dataset = tf.data.TFRecordDataset(
            each_record, compression_type=compression_type)
        if dataset is None:
            dataset = new_dataset
        else:
            dataset = dataset.concatenate(new_dataset)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    auto_tune = tf.data.experimental.AUTOTUNE
    dataset = dataset.batch(batch_size=batch_size,
                            drop_remainder=drop_remainder).prefetch(auto_tune)
    dataset = dataset.map(lambda example: decode_tfrecord(example, batch_size))
    return dataset


def plot_comparison_figure(input_dir: str, channels: list, scenarios: list, slice_num: int, output_dir: str):
    plt.figure()
    for channel_idx, channel in enumerate(channels):
        ground_truth_scenario = ''
        for i in range(4):
            if i == channel_idx:
                ground_truth_scenario += '1'
            else:
                ground_truth_scenario += '0'
        ground_truth_dir = os.path.join(input_dir, ground_truth_scenario)
        ground_truth_path = os.path.join(
            ground_truth_dir, "%s.nii.gz" % channel)
        ground_truth_arr = load_nii(ground_truth_path)[slice_num]
        each_img_shape = ground_truth_arr.shape
        img_shape = (ground_truth_arr.shape[0], ground_truth_arr.shape[1] * 8)
        channel_img = np.empty(shape=img_shape, dtype="float32")
        channel_img[:, 0:each_img_shape[1]] = ground_truth_arr
        title_name = ' ground truth '
        for scenario_idx, each_scenario in enumerate(scenarios[channel_idx]):
            title_name += '     %s     ' % each_scenario
            img_dir = os.path.join(input_dir, each_scenario)
            img_path = os.path.join(img_dir, "%s.nii.gz" % channel)
            img = load_nii(img_path)[slice_num]
            channel_img[:, each_img_shape[0] * (scenario_idx + 1): each_img_shape[0] * (
                scenario_idx + 1) + each_img_shape[1]] = img
        plt.subplot(4, 1, channel_idx + 1)
        plt.xticks([])
        plt.yticks([])
        # plt.title(title_name)
        plt.imshow(channel_img, cmap='gray')
    save_path = os.path.join(output_dir, "%s.png" % os.path.split(input_dir)[-1])
    plt.savefig(save_path, dpi=1000)
    plt.show()


if __name__ == '__main__':
    tfrecords_train = [
        r"E:\my_files\programmes\python\BRATS2018_normalized\group0_standardized.tfrecord",
        r"E:\my_files\programmes\python\BRATS2018_normalized\group1_standardized.tfrecord",
        r"E:\my_files\programmes\python\BRATS2018_normalized\group2_standardized.tfrecord",
        r"E:\my_files\programmes\python\BRATS2018_normalized\group3_standardized.tfrecord",
    ]
    tfrecords_val = [
        r"E:\my_files\programmes\python\BRATS2018_normalized\group4_standardized.tfrecord"
    ]
    # print(get_dataset(tfrecords_train, 'ZLIB', 8, True, True, 2000))
    plot_comparison_figure(
        input_dir=r'E:\my_files\programmes\python\mri_gan_output\comb4_cropped_demo\Brats18_CBICA_AAP_1',
        channels=constants_and_tools.CHANNELS,
        output_dir=r'E:\my_files\programmes\python\mri_gan_output\comb4_cropped_demo\Brats18_CBICA_AAP_1',
        scenarios=[
            ['0111', '0110', '0101', '0100', '0011', '0010', '0001'],
            ['1011', '1010', '1001', '1000', '0011', '0010', '0001'],
            ['1101', '1100', '1001', '1000', '0101', '0100', '0001'],
            ['1110', '1100', '1010', '1000', '0110', '0100', '0010'],
        ],
        slice_num=78
    )
