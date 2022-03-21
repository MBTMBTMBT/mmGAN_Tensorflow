import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import GradientTape
import tqdm
from mm_gan.constants_and_tools import make_dir
from tensorflow.train import Checkpoint, CheckpointManager
import sys

import mm_gan.models_two_channels
import mm_gan.io_helpers

TFRECORDS_TRAIN = [
    r"E:\my_files\programmes\python\BRATS2018_normalized\group0_standardized.tfrecord",
    r"E:\my_files\programmes\python\BRATS2018_normalized\group1_standardized.tfrecord",
    r"E:\my_files\programmes\python\BRATS2018_normalized\group2_standardized.tfrecord",
    r"E:\my_files\programmes\python\BRATS2018_normalized\group3_standardized.tfrecord",
]
TFRECORDS_VAL = [
    r"E:\my_files\programmes\python\BRATS2018_normalized\group4_standardized.tfrecord"
]


def train_without_discriminator(session_name: str, output_dir: str,
                                tfrecords_train: list, tfrecords_val: list,
                                batch_size_train: int, batch_size_val: int,
                                learning_rate: float, beta_1: float, beta_2: float,
                                lambda_param: float, epochs: int):
    # make dir for output dir
    output_dir = os.path.join(output_dir, session_name)
    make_dir(output_dir)

    log_dir = os.path.join(output_dir, 'logs')
    make_dir(log_dir)

    json_dir = os.path.join(output_dir, 'jsons')
    make_dir(json_dir)

    # get loggers
    train_logger = logging.getLogger(name=session_name + "_train")
    train_logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)  # stderr output to console
    log_path_train = os.path.join(log_dir, session_name + "_train" + ".txt")
    file_handler = logging.FileHandler(log_path_train, mode='a')
    formatter = logging.Formatter('%(asctime)s: %(message)s', "%y%b%d-%H:%M:%S")
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    train_logger.addHandler(stream_handler)
    train_logger.addHandler(file_handler)

    train_dataset = None
    val_dataset = None
    for each_record in tfrecords_train:
        dataset = tf.data.TFRecordDataset(each_record, compression_type='ZLIB')
        if train_dataset is None:
            train_dataset = dataset
        else:
            train_dataset = train_dataset.concatenate(dataset)
    for each_record in tfrecords_val:
        dataset = tf.data.TFRecordDataset(each_record, compression_type='ZLIB')
        if val_dataset is None:
            val_dataset = dataset
        else:
            val_dataset = val_dataset.concatenate(dataset)

    auto_tune = tf.data.experimental.AUTOTUNE  # shuffle(buffer_size=2000).
    train_dataset = train_dataset.batch(batch_size=batch_size_train, drop_remainder=True).prefetch(buffer_size=auto_tune)
    train_dataset = train_dataset.map(lambda example: io_helpers.decode_tfrecord(example, batch_size_train))
    val_dataset = val_dataset.shuffle(buffer_size=2000).batch(batch_size=batch_size_val,
                                                              drop_remainder=True).prefetch(buffer_size=auto_tune)
    val_dataset = val_dataset.map(lambda example: io_helpers.decode_tfrecord(example, batch_size_val))

    generator = models_two_channels.get_generator_unet()
    # get optimizers
    optimizer_g = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

    mae = tf.keras.losses.MeanAbsoluteError()
    mse = tf.keras.losses.MeanSquaredError()

    # use checkpoints
    checkpoint_dir = os.path.join(output_dir, "saved_checkpoints")
    make_dir(checkpoint_dir)
    # checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
    checkpoint = Checkpoint(generator_optimizer=optimizer_g, generator=generator)
    checkpoint_manager = CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=epochs)

    for epoch_idx in range(epochs):
        loss = 0
        count = 0
        for names, t1s, t2s, t1ces, flairs in tqdm.tqdm(train_dataset, desc="epoch %d" % epoch_idx):
            loss = train_without_discriminator_step(generator, t1s, t2s, t1ces, flairs, optimizer_g, mae, mse, count)
            count += 1
        else:
            checkpoint_manager.save(checkpoint_number=epoch_idx)
            train_logger.info("epoch %d, mae: %s" % (epoch_idx, str(loss.numpy().item())))


# @tf.function
def train_without_discriminator_step(generator, t1_batch, t2_batch, t1ce_batch, flair_batch, optimizer, mae, mse, count):
    # init tape
    with GradientTape() as g_tape:
        batch_x = tf.concat([t1_batch, t2_batch], axis=1)
        batch_y = tf.concat([t1ce_batch, flair_batch], axis=1)

        batch_out = generator(batch_x, training=True)
        if count == 0:
            plt.figure()
            plt.imshow(batch_x[0, 0, ...])
            plt.title("t1")
            plt.figure()
            plt.imshow(batch_x[0, 1, ...])
            plt.title("t2")
            plt.figure()
            plt.imshow(batch_y[0, 0, ...])
            plt.title("t1ce")
            plt.figure()
            plt.imshow(batch_y[0, 1, ...])
            plt.title("flair")
            plt.figure()
            plt.imshow(batch_out[0, 0, ...])
            plt.title("t1ce_fake")
            plt.figure()
            plt.imshow(batch_out[0, 1, ...])
            plt.title("flair_fake")
            plt.show()

        loss_pixel = mae(batch_out, batch_y)
        # loss_pixel = mse(batch_out, batch_y)

    # apply gradients
    gradients_of_generator = g_tape.gradient(loss_pixel, generator.trainable_variables)
    optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    return loss_pixel


@tf.function
def validation(generator, dataset, mse):
    pass


@tf.function
def test(generator, name, t1_batch, t2_batch, t1ce_batch, flair_batch):
    batch_x = tf.concat([t1_batch, t2_batch], axis=1)
    batch_y = tf.concat([t1ce_batch, flair_batch], axis=1)
    batch_out = generator(batch_x, training=True)


if __name__ == '__main__':
    train_without_discriminator(session_name="test_2_to_2_02",
                                output_dir=r"E:\my_files\programmes\python\mri_gan_output",
                                tfrecords_train=TFRECORDS_TRAIN, tfrecords_val=TFRECORDS_VAL,
                                batch_size_train=8, batch_size_val=32,
                                learning_rate=0.0002, beta_1=0.5, beta_2=0.999, lambda_param=0.9, epochs=60)
