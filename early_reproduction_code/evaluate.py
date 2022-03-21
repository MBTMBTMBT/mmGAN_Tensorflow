import matplotlib.pyplot as plt
import pickle
import os
import re
from settings import *


def evaluate(results: dict) -> (list, list, list):
    mse_list = []
    psnr_list = []
    ssim_list = []
    rst = {'mst': mse_list, 'psnr': psnr_list, 'ssim': ssim_list}
    LOGGER.info("results:")
    LOGGER.info('\t\t\tmse\t\t\tpsnr\t\tssim')
    for k in range(len(results)):
        mse = results[k]['mse']
        psnr = results[k]['psnr']
        ssim = results[k]['ssim']
        mse_list.append(mse)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        LOGGER.info("epoch %2d\t%f\t%f\t%f" % (k + 1, mse, psnr, ssim))
    return mse_list, psnr_list, ssim_list


def get_results_list(results_dir: str) -> dict:
    results_dict = {}
    dir_list = os.listdir(results_dir)
    for num in range(0, RUN_EPOCHS):
        # print(num)
        try:
            with open(os.path.join(results_dir, 'result_dict_test_epoch_%d.pkl' % num), 'rb') as f:
                rst = pickle.load(f)
                # print(rst)
                # print(rst['mean']['mse'])
                results_dict[num] = {'mse': None, 'psnr': None, 'ssim': None}
                results_dict[num]['mse'] = rst['mean']['mse']
                results_dict[num]['psnr'] = rst['mean']['psnr']
                results_dict[num]['ssim'] = rst['mean']['ssim']
        except FileNotFoundError:
            break
    return results_dict


def draw_lines(mse_list: list, psnr_list: list, ssim_list: list):
    plt.figure(figsize=(16, 6))
    plt.title('mean results')
    im1 = plt.subplot(1, 3, 1)
    im2 = plt.subplot(1, 3, 2)
    im3 = plt.subplot(1, 3, 3)

    plt.sca(im1)
    plt.plot(range(1, len(mse_list) + 1), mse_list, color='red', label='mse')
    plt.xlabel('epoch')
    plt.ylabel('mse')

    plt.sca(im2)
    plt.plot(range(1, len(mse_list) + 1), psnr_list, color='blue', label='psnr')
    plt.xlabel('epoch')
    plt.ylabel('psnr')

    plt.sca(im3)
    plt.plot(range(1, len(mse_list) + 1), ssim_list, color='green', label='ssim')
    plt.xlabel('epoch')
    plt.ylabel('ssim')
    plt.show()


if __name__ == '__main__':
    mse_list, psnr_list, ssim_list = evaluate(get_results_list(os.path.join(DATASET_PATH, 'hgg')))
    draw_lines(mse_list, psnr_list, ssim_list)
