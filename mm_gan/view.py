import matplotlib.pyplot as plt
import numpy as np
try:
    import mm_gan.io_helpers as io_helpers
except ModuleNotFoundError:
    import io_helpers
import os


def plot_img(file_path: str, slice_num: int, name=''):
    img_arr = io_helpers.load_nii(file_path)
    print("shape: ", img_arr.shape)
    print("max: ", np.max(img_arr), " min: ", np.min(img_arr))
    print("mean: ", np.mean(img_arr), " std: ", np.std(img_arr))
    plot_slice = img_arr[slice_num]
    zero_voxel = plot_slice[0, 0]
    plot_slice[plot_slice < zero_voxel] = zero_voxel
    plt.imshow(plot_slice, cmap='gray')
    plt.title(name)
    plt.show()


def plot_patient(patient_dir: str, channels: list, slice_num: int, name=''):
    plt.figure()
    plt.suptitle(name)
    for idx, each_channel in enumerate(channels):
        img_path = os.path.join(patient_dir, "%s.nii.gz" % each_channel)
        img_arr = io_helpers.load_nii(img_path)
        plot_slice = img_arr[slice_num]
        # zero_voxel = plot_slice[0, 0]
        # plot_slice[plot_slice < zero_voxel] = zero_voxel
        plt.subplot(2, 4, idx + 1)
        plt.title(each_channel)
        plt.imshow(plot_slice, cmap='gray')
        print(each_channel)
        print("shape: ", img_arr.shape)
        print("max: ", np.max(img_arr), " min: ", np.min(img_arr))
        print("mean: ", np.mean(img_arr), " std: ", np.std(img_arr))

        img_1d = np.reshape(
            plot_slice, (plot_slice.shape[0] * plot_slice.shape[1]))
        plt.subplot(2, 4, idx + 5)
        # plt.title(each_channel)
        plt.plot(img_1d)
    plt.show()


def img_to_img_compare(img_a_path: str, img_b_path: str, slice_num_1: int, slice_num_2: int, title1='', title2=''):
    img_a = io_helpers.load_nii(img_a_path)[slice_num_1]
    img_b = io_helpers.load_nii(img_b_path)[slice_num_2]
    assert len(img_a.shape) == len(img_b.shape)
    assert img_a.shape[0] == img_b.shape[0]
    img_2d = np.empty(shape=(img_a.shape[0], img_a.shape[1] + img_b.shape[1]))
    img_2d[:, 0:img_a.shape[1]] = img_a
    img_2d[:, img_b.shape[1]:img_b.shape[1]*2] = img_b
    img_1d_a = np.reshape(img_a, (img_a.shape[0] * img_a.shape[1]))
    img_1d_b = np.reshape(img_b, (img_b.shape[0] * img_b.shape[1]))
    img_1d = np.empty(shape=(img_1d_a.shape[0] + img_1d_b.shape[0]))
    img_1d[0:img_1d_a.shape[0]] = img_1d_a
    img_1d[img_1d_b.shape[0]:img_1d_b.shape[0]*2] = img_1d_b
    plt.figure()
    plt.imshow(img_2d, cmap='gray')
    plt.title("%s                                    %s" % (title1, title2))
    plt.figure()
    plt.plot(img_1d)
    plt.title("%s                                    %s" % (title1, title2))
    plt.show()


if __name__ == '__main__':
    plot_img(r'E:\my_files\programmes\python\BRATS2018_normalized\fold4\Brats18_CBICA_AAP_1\t2_standardized.nii.gz',
            89, 'Brats18_CBICA_AAP_1')
    '''plot_patient(r'E:\my_files\programmes\python\BRATS2018\Training\HGG\Brats18_TCIA01_186_1',
                 ['Brats18_TCIA01_186_1_t1', 
                 'Brats18_TCIA01_186_1_t2', 
                 'Brats18_TCIA01_186_1_t1ce', 
                 'Brats18_TCIA01_186_1_flair'],
                 89, 'Brats18_TCIA01_186_1')'''
    '''plot_patient(r'E:\my_files\programmes\python\BRATS2018_normalized\fold4\Brats18_TCIA01_186_1',
                 ['t1_standardized', 't2_standardized', 't1ce_standardized', 'flair_standardized'],
                 89, 'Brats18_TCIA01_186_1')'''
    '''plot_patient(r'E:\my_files\programmes\python\mri_gan_output\comb0\test-epoch19\Brats18_TCIA01_186_1\1101',
                 ['t1', 't2', 't1ce', 'flair'], 79, 'Brats18_TCIA01_186_1')'''
    '''plot_patient(r'E:\my_files\programmes\python\BRATS2018\Training\HGG\Brats18_CBICA_AAP_1',
                 ['Brats18_CBICA_AAP_1_t1', 
                 'Brats18_CBICA_AAP_1_t2', 
                 'Brats18_CBICA_AAP_1_t1ce', 
                 'Brats18_CBICA_AAP_1_flair'],
                 89, 'Brats18_CBICA_AAP_1')'''
    '''plot_patient(r'E:\my_files\programmes\python\BRATS2018_normalized\fold4\Brats18_CBICA_AAP_1',
                 ['t1_standardized', 't2_standardized', 't1ce_standardized', 'flair_standardized'],
                 89, 'Brats18_CBICA_AAP_1')
    plot_patient(r'E:\my_files\programmes\python\mri_gan_output\comb0\test-epoch59\Brats18_CBICA_AAP_1\1101',
                 ['t1', 't2', 't1ce', 'flair'], 79, 'Brats18_CBICA_AAP_1')'''
    '''plot_patient(r'E:\my_files\programmes\python\BRATS2018_normalized\fold4\Brats18_TCIA01_390_1',
                 ['t1_standardized', 't2_standardized', 't1ce_standardized', 'flair_standardized'],
                 89, 'Brats18_TCIA01_390_1')'''
    '''plot_patient(r'E:\my_files\programmes\python\mri_gan_output\test_07\test-epoch59\Brats18_TCIA01_390_1\1101',
                 ['t1', 't2', 't1ce', 'flair'], 79, 'Brats18_TCIA01_390_1')
    Brats18_TCIA02_171_1'''
    '''plot_patient(r'E:\my_files\programmes\python\BRATS2018\Training\HGG\Brats18_TCIA02_171_1',
                 ['Brats18_TCIA02_171_1_t1', 
                 'Brats18_TCIA02_171_1_t2', 
                 'Brats18_TCIA02_171_1_t1ce', 
                 'Brats18_TCIA02_171_1_flair'],
                 75, 'Brats18_TCIA02_171_1')'''
    '''plot_patient(r'E:\my_files\programmes\python\BRATS2018_normalized\fold4\Brats18_CBICA_AAP_1',
                 ['t1_standardized', 't2_standardized', 't1ce_standardized', 'flair_standardized'],
                 20, 'Brats18_TCIA02_171_1')'''
    '''plot_patient(r'E:\my_files\programmes\python\BRATS2018_normalized\fold4\Brats18_TCIA02_171_1',
                 ['t1_standardized', 't2_standardized', 't1ce_standardized', 'flair_standardized'],
                 40, 'Brats18_TCIA02_171_1')'''
    '''plot_patient(r'E:\my_files\programmes\python\BRATS2018_normalized\fold4\Brats18_TCIA01_186_1',
                 ['t1_standardized', 't2_standardized', 't1ce_standardized', 'flair_standardized'],
                 120, 'Brats18_TCIA02_171_1')'''
    '''plot_patient(r'E:\my_files\programmes\python\mri_gan_output\test_07\test-epoch59\Brats18_TCIA02_171_1\1101',
                 ['t1', 't2', 't1ce', 'flair'], 65, 'Brats18_TCIA02_171_1')'''
    '''
    img_to_img_compare(
        img_a_path=r'E:\my_files\programmes\python\BRATS2018_normalized\fold4\Brats18_CBICA_AAP_1\t1ce_standardized.nii.gz',
        img_b_path=r'E:\my_files\programmes\python\mri_gan_output\comb0\test-epoch59\Brats18_CBICA_AAP_1\1101\t1ce.nii.gz',
        slice_num_1=91,
        slice_num_2=79,
        title1="gt",
        title2='1101'
    )
    img_to_img_compare(
        img_a_path=r'E:\my_files\programmes\python\BRATS2018_normalized\fold4\Brats18_CBICA_AAP_1\t1ce_standardized.nii.gz',
        img_b_path=r'E:\my_files\programmes\python\mri_gan_output\comb0\test-epoch59\Brats18_CBICA_AAP_1\1100\t1ce.nii.gz',
        slice_num_1=91,
        slice_num_2=79,
        title1="gt",
        title2='1100'
    )
    img_to_img_compare(
        img_a_path=r'E:\my_files\programmes\python\BRATS2018_normalized\fold4\Brats18_CBICA_AAP_1\t1ce_standardized.nii.gz',
        img_b_path=r'E:\my_files\programmes\python\mri_gan_output\comb0\test-epoch59\Brats18_CBICA_AAP_1\1001\t1ce.nii.gz',
        slice_num_1=91,
        slice_num_2=79,
        title1="gt",
        title2='1001'
    )
    img_to_img_compare(
        img_a_path=r'E:\my_files\programmes\python\BRATS2018_normalized\fold4\Brats18_CBICA_AAP_1\t1ce_standardized.nii.gz',
        img_b_path=r'E:\my_files\programmes\python\mri_gan_output\comb0\test-epoch59\Brats18_CBICA_AAP_1\0101\t1ce.nii.gz',
        slice_num_1=91,
        slice_num_2=79,
        title1="gt",
        title2='0101'
    )
    img_to_img_compare(
        img_a_path=r'E:\my_files\programmes\python\BRATS2018_normalized\fold4\Brats18_CBICA_AAP_1\t1ce_standardized.nii.gz',
        img_b_path=r'E:\my_files\programmes\python\mri_gan_output\comb0\test-epoch59\Brats18_CBICA_AAP_1\1000\t1ce.nii.gz',
        slice_num_1=91,
        slice_num_2=79,
        title1="gt",
        title2='1000'
    )
    img_to_img_compare(
        img_a_path=r'E:\my_files\programmes\python\BRATS2018_normalized\fold4\Brats18_CBICA_AAP_1\t1ce_standardized.nii.gz',
        img_b_path=r'E:\my_files\programmes\python\mri_gan_output\comb0\test-epoch59\Brats18_CBICA_AAP_1\0100\t1ce.nii.gz',
        slice_num_1=91,
        slice_num_2=79,
        title1="gt",
        title2='0100'
    )
    img_to_img_compare(
        img_a_path=r'E:\my_files\programmes\python\BRATS2018_normalized\fold4\Brats18_CBICA_AAP_1\t1ce_standardized.nii.gz',
        img_b_path=r'E:\my_files\programmes\python\mri_gan_output\comb0\test-epoch59\Brats18_CBICA_AAP_1\0001\t1ce.nii.gz',
        slice_num_1=91,
        slice_num_2=79,
        title1="gt",
        title2='0001'
    )
    '''
    
    '''
    plot_patient(r'E:\my_files\programmes\python\BRATS2018_normalized\fold4\Brats18_TCIA01_425_1',
                 ['t1_standardized', 't2_standardized', 't1ce_standardized', 'flair_standardized'],
                 45, 'Brats18_TCIA01_425_1')
                 '''
