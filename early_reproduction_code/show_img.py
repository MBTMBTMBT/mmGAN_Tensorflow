import SimpleITK as sitk
from matplotlib import pyplot as plt


def showNii(img):
    plt.imshow(img[55, :, :], cmap='gray')
    plt.show()
    # for i in range(img.shape[0]):
    #     plt.imshow(img[i, :, :], cmap='gray')
    #     plt.show()


def show_hdf5(img):
    plt.imshow(img, cmap='gray')
    plt.show()


def show_img(img):
    plt.imshow(img, cmap='gray')
    plt.show()


if __name__ == '__main__':
    itk_img = sitk.ReadImage(r'D:\python\new_mmgan\BRATS2018\Training\HGG\Brats18_CBICA_ABB_1\Brats18_CBICA_ABB_1_t1ce.nii.gz')
    img = sitk.GetArrayFromImage(itk_img)
    print(img.shape)
    showNii(img)
