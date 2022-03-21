import re
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing
from mm_gan.io_helpers import load_nii, write_nii


def find_names_and_dice_values(txt_path: str) -> tuple:
    file = open(txt_path, 'r')
    content = file.read()
    names = re.findall(r"predictions\\(.*?)\\predicted_Segm.nii.gz", content)
    # dice = re.findall(r"(\d+\.\d+)",re.search(r"(?<=DICE1=.).+(?=] DICE2)",content).group())
    dices = re.findall(r"DICE1=(.*?)DICE2", content)
    dice_values = []
    average = []
    for idx, d in enumerate(dices):
        dice_strs = re.findall(r"\d+\.?\d*", d)
        dice_floats = []
        for each in dice_strs:
            dice_floats.append(float(each))
        if idx < len(dices) - 1:
            dice_values.append(dice_floats)
        else:
            average = dice_floats  # the last results are means
    dice_values_arr = np.array(dice_values)
    std = np.std(dice_values, axis=0).tolist()
    return names, dice_values, average, std


def conclude_tests(txt_paths: list, scenarios: list) -> dict:
    dice_means = {}
    print("scen\t\tclass0\tclass1\tclass2\tclass4\n======================================")
    for each_file, scenario in zip(txt_paths, scenarios):
        _, dice_values, dice_mean, dice_std = find_names_and_dice_values(each_file)
        dice_means[scenario] = dice_mean
        print("%s-mean\t%.4f\t%.4f\t%.4f\t%.4f" % (scenario, dice_mean[0], dice_mean[1], dice_mean[2], dice_mean[3]))
        print("%s-std\t%.4f\t%.4f\t%.4f\t%.4f" % (scenario, dice_std[0], dice_std[1], dice_std[2], dice_std[3]))
    print("scen\t\tclass0\tclass1\tclass2\tclass4\n======================================")
    for each_file, scenario in zip(txt_paths, scenarios):
        _, dice_values, dice_mean, dice_std = find_names_and_dice_values(each_file)
        dice_means[scenario] = dice_mean
        print("%s-mean\t%.4f\t%.4f\t%.4f\t%.4f" % (scenario, dice_mean[0], dice_mean[1], dice_mean[2], dice_mean[3]))
    print("scen\t\tclass0\tclass1\tclass2\tclass4\n======================================")
    for each_file, scenario in zip(txt_paths, scenarios):
        _, dice_values, dice_mean, dice_std = find_names_and_dice_values(each_file)
        dice_means[scenario] = dice_mean
        print("%s-std\t%.4f\t%.4f\t%.4f\t%.4f" % (scenario, dice_std[0], dice_std[1], dice_std[2], dice_std[3]))
    return dice_means


def plot_comparison_sge_masks(input_paths: list, slice_num: int, output_path: str):
    plt.figure()
    assert len(input_paths) == 16
    gt = load_nii(input_paths[0])[slice_num]
    img_a = np.zeros(shape=(gt.shape[0], gt.shape[1] * 8), dtype='float32')
    img_b = np.zeros(shape=(gt.shape[0], gt.shape[1] * 8), dtype='float32')
    for i in range(0, 8):
        img_a[:, i * gt.shape[1]: (i + 1) * gt.shape[1]] = load_nii(input_paths[i])[slice_num]
    plt.subplot(2, 1, 1)
    plt.imshow(img_a)
    plt.xticks([])
    plt.yticks([])
    for i in range(8, 16):
        img_b[:, (i - 8) * gt.shape[1]: (i - 8 + 1) * gt.shape[1]] = load_nii(input_paths[i])[slice_num]
    plt.subplot(2, 1, 2)
    plt.imshow(img_b)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(output_path, dpi=1000)
    plt.show()


def zeros_nii_from(nii_path: str, output_dir: str):
    nii_arr = load_nii(nii_path)
    ones_arr = np.ones_like(nii_arr)
    print(ones_arr.shape)
    zeros_nii = ones_arr * nii_arr[0, 0, 0]
    write_nii(zeros_nii, os.path.join(output_dir, "zero_channel.nii.gz"))
    return zeros_nii


def compare_volume(segm_ground_truth_nii_paths: list, segm_generated_nii_paths: list, labels: list, show_img=False):
    assert len(segm_ground_truth_nii_paths) == len(segm_generated_nii_paths)
    gt_iter = segm_ground_truth_nii_paths.__iter__()
    gnt_iter = segm_generated_nii_paths.__iter__()
    gt_volumes = {}
    gnt_volumes = {}
    poly_vals = {}
    for lb in labels:
        gt_volumes[lb] = []
        gnt_volumes[lb] = []
    '''
    for _ in tqdm.tqdm(
            iterable=range(len(segm_ground_truth_nii_paths)),
            total=len(segm_ground_truth_nii_paths)
    ):
    '''
    for _ in range(len(segm_ground_truth_nii_paths)):
        gt_path = gt_iter.__next__()
        gnt_path = gnt_iter.__next__()
        gt_mat = load_nii(gt_path)
        gnt_mat = load_nii(gnt_path)
        for lb in labels:
            gt_with_label = np.isin(gt_mat, [lb]).astype(dtype="int8")
            gnt_with_label = np.isin(gnt_mat, [lb]).astype(dtype="int8")
            gt_label_volume = np.sum(gt_with_label, dtype=int)
            gnt_label_volume = np.sum(gnt_with_label, dtype=int)
            gt_volumes[lb].append(gt_label_volume)
            gnt_volumes[lb].append(gnt_label_volume)
    outliers = {}
    best_performances = {}
    medium_performances = {}
    for idx, lb in enumerate(labels):
        gt_volumes[lb] = np.array(gt_volumes[lb])
        gnt_volumes[lb] = np.array(gnt_volumes[lb])

        # locate outliers
        error_rate = gt_volumes[lb] / gnt_volumes[lb]
        error_rate[~np.isfinite(error_rate)] = 0
        outlier_idx = error_rate.tolist().index(max(error_rate))
        outlier_path = segm_generated_nii_paths[outlier_idx]
        outliers[lb] = outlier_path

        # locate best performance
        error_rate = np.abs(error_rate - 1)
        best_idx = error_rate.tolist().index(min(error_rate))
        best_path = segm_generated_nii_paths[best_idx]
        best_performances[lb] = best_path

        # locate midium performance
        error_rate = np.abs(error_rate - 0.5)
        mid_idx = error_rate.tolist().index(min(error_rate))
        mid_path = segm_generated_nii_paths[mid_idx]
        medium_performances[lb] = mid_path

        poly = np.polyfit(gt_volumes[lb], gnt_volumes[lb], deg=1)
        poly_val = np.polyval(poly, gt_volumes[lb])
        poly_vals[lb] = poly_val
        plt.figure(idx+1)
        plt.plot(gt_volumes[lb], gnt_volumes[lb], 'o')
        plt.plot(gt_volumes[lb], poly_val)
        plt.title("label %d" % lb)
        plt.xlabel("ground truth")
        plt.ylabel("deepmedic")
        if show_img:
            plt.imshow()
    return gt_volumes, gnt_volumes, poly_vals, outliers, best_performances, medium_performances


def compare_volume_multi_processing(para_dict: dict) -> dict:
    print("working on scenario: %s" % para_dict['scenario'])
    gt_dirs = para_dict['gt_dirs']
    gnt_dirs = para_dict['gnt_dirs']
    labels = para_dict['labels']
    gt_volumes, gnt_volumes, poly_vals, outliers, best_performances, medium_performances \
        = compare_volume(gt_dirs, gnt_dirs, labels)
    rt_dict = {
        'gt_volumes': gt_volumes,
        'gnt_volumes': gnt_volumes,
        'poly_vals': poly_vals,
        'scenario': para_dict['scenario'],
        'outliers': outliers,
        'best_performances': best_performances,
        'medium_performances': medium_performances,
    }
    return rt_dict


def compare_volume_with_dir(top_dir: str, gt_nii_name: str, gnt_nii_name: str, scenarios: list, labels: list, workers: int):
    gt_dir = os.path.join(top_dir, 'gt')
    patient_names = os.listdir(gt_dir)
    gt_dirs = []
    for each_name in patient_names:
        gt_patient_dir = os.path.join(gt_dir, each_name)
        gt_dirs.append(os.path.join(gt_patient_dir, gt_nii_name))
    gnt_dir_dict = {}
    para_dicts = []
    for each_scenario in scenarios:
        gnt_dir_dict[each_scenario] = []
        scen_dir = os.path.join(top_dir, each_scenario)
        for each_name in patient_names:
            scen_patient_dir = os.path.join(scen_dir, each_name)
            gnt_dir_dict[each_scenario].append(os.path.join(scen_patient_dir, gnt_nii_name))
        # print("working on scenario: %s" % each_scenario)
        para_dict = {
            'gt_dirs': gt_dirs,
            'gnt_dirs': gnt_dir_dict[each_scenario],
            'labels': labels,
            'scenario': each_scenario
        }
        para_dicts.append(para_dict)
        # gt_volumes, gnt_volumes, gnt_poly_vals = compare_volume(gt_dirs, gnt_dir_dict[each_scenario], [0, 1, 2, 4])
    pool = multiprocessing.Pool(workers)
    rt_dicts = pool.map(compare_volume_multi_processing, para_dicts)
    # print(rt_dicts)
    for i, each_rt_dict in enumerate(rt_dicts):
        for j, lb in enumerate(labels):
            plt.figure(i*len(labels) + j + 1)
            plt.plot(each_rt_dict['gt_volumes'][lb], each_rt_dict['gnt_volumes'][lb], 'o')
            plt.plot(each_rt_dict['gt_volumes'][lb], each_rt_dict['poly_vals'][lb])
            plt.title("scenario %s label %d" % (each_rt_dict['scenario'], lb))
            plt.xlabel("ground truth")
            plt.ylabel("deepmedic")
            plt.savefig(os.path.join(top_dir, 'scenario %s label %d.png' % (each_rt_dict['scenario'], lb)),
                        bbox_inches='tight')
            print("scenario %s label %d outlier %s" % (each_rt_dict['scenario'], lb, each_rt_dict['outliers'][lb]))
            print("scenario %s label %d best performances %s" % (each_rt_dict['scenario'], lb, each_rt_dict['best_performances'][lb]))
            print("scenario %s label %d medium performances %s" % (each_rt_dict['scenario'], lb, each_rt_dict['medium_performances'][lb]))
    plt.close(fig='all')


def calculate_dice(pred_seg, gt_lbl):
    union_correct = pred_seg * gt_lbl
    tp_num = np.sum(union_correct)
    gt_pos_num = np.sum(gt_lbl)
    dice = (2.0 * tp_num) / (np.sum(pred_seg) + gt_pos_num) if gt_pos_num != 0 else -1
    return dice


def compute_dice_for_et_tc_wt(pred_seg: np.ndarray, gt_lbl:np.ndarray):
    rint_pred = np.rint(pred_seg)
    rint_gt = np.rint(gt_lbl)
    lbl4_pred = rint_pred == 4
    lbl4_gt = rint_gt == 4
    lbl2_pred = rint_pred == 2
    lbl2_gt = rint_gt == 2
    lbl1_pred = rint_pred == 1
    lbl1_gt = rint_gt == 1
    
    # enhancing tumor
    et_pred = lbl4_pred
    et_gt = lbl4_gt

    # tumor core
    ct_pred = np.logical_or(lbl4_pred, lbl1_pred)
    ct_gt = np.logical_or(lbl4_gt, lbl1_gt)

    # whole tumor
    wt_pred = np.logical_or(ct_pred, lbl2_pred)
    wt_gt = np.logical_or(ct_gt, lbl2_gt)

    # compute dice
    et_dice = calculate_dice(et_pred, et_gt)
    ct_dice = calculate_dice(ct_pred, ct_gt)
    wt_dice = calculate_dice(wt_pred, wt_gt)

    return et_dice, ct_dice, wt_dice


def compute_dice_with_pred_masks(pred_patients_dir: str, gt_patients_dir: str, patient_nii_name: str, gt_nii_name: str):
    patient_names = os.listdir(pred_patients_dir)
    pred_patient_paths = []
    gt_paths = []
    for each_name in patient_names:
        pred_patient_dir = os.path.join(pred_patients_dir, each_name)
        pred_patient_paths.append(os.path.join(pred_patient_dir, patient_nii_name))
        gt_patient_dir = os.path.join(gt_patients_dir, each_name)
        gt_paths.append(os.path.join(gt_patient_dir, gt_nii_name))
    sum_et = 0
    sum_ct = 0
    sum_wt = 0
    count = 0
    for idx in range(len(pred_patient_paths)):
        pred_lbls = load_nii(pred_patient_paths[idx])
        gt_lbls = load_nii(gt_paths[idx])[12:148]
        et, ct, wt = compute_dice_for_et_tc_wt(pred_lbls, gt_lbls)
        sum_et += et
        sum_ct += ct
        sum_wt += wt
        count += 1
    et_avg = sum_et / count
    ct_avg = sum_ct / count
    wt_avg = sum_wt / count
    # print(et_avg, ct_avg, wt_avg)
    return et_avg, ct_avg, wt_avg


if __name__ == '__main__':
    txts = [
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0001\logs\test_comb0_0001.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0010\logs\test_comb0_0010.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0100\logs\test_comb0_0100.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1000\logs\test_comb0_1000.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0011\logs\test_comb0_0011.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0101\logs\test_comb0_0101.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0110\logs\test_comb0_0110.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1001\logs\test_comb0_1001.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1010\logs\test_comb0_1010.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1100\logs\test_comb0_1100.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0111\logs\test_comb0_0111.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1011\logs\test_comb0_1011.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1101\logs\test_comb0_1101.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1110\logs\test_comb0_1110.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1111\logs\test_comb0_1111.txt",
    ]
    txts_ref = [
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0001_ref\logs\test_comb0_0001.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0010_ref\logs\test_comb0_0010.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0100_ref\logs\test_comb0_0100.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1000_ref\logs\test_comb0_1000.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0011_ref\logs\test_comb0_0011.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0101_ref\logs\test_comb0_0101.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0110_ref\logs\test_comb0_0110.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1001_ref\logs\test_comb0_1001.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1010_ref\logs\test_comb0_1010.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1100_ref\logs\test_comb0_1100.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0111_ref\logs\test_comb0_0111.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1011_ref\logs\test_comb0_1011.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1101_ref\logs\test_comb0_1101.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1110_ref\logs\test_comb0_1110.txt",
        r"E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1111\logs\test_comb0_1111.txt",
    ]
    scenarios = ['0001', '0010', '0100', '1000',
                 '0011', '0101', '0110', '1001', '1010', '1100',
                 '0111', '1011', '1101', '1110', '1111']

    segs = [
        r'E:\my_files\programmes\python\BRATS2018_normalized\fold4\Brats18_CBICA_AAP_1\seg_preprocessed.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1111\predictions\test_comb0_1111\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1110\predictions\test_comb0_1110\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1101\predictions\test_comb0_1101\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1011\predictions\test_comb0_1011\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0111\predictions\test_comb0_0111\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1100\predictions\test_comb0_1100\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1010\predictions\test_comb0_1010\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1001\predictions\test_comb0_1001\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0110\predictions\test_comb0_0110\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0101\predictions\test_comb0_0101\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0011\predictions\test_comb0_0011\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1000\predictions\test_comb0_1000\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0100\predictions\test_comb0_0100\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0010\predictions\test_comb0_0010\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0001\predictions\test_comb0_0001\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
    ]

    segs = [
        r'E:\my_files\programmes\python\BRATS2018_normalized\fold4\Brats18_TCIA01_425_1\seg_preprocessed.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1111\Brats18_TCIA01_425_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1110\Brats18_TCIA01_425_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1101\Brats18_TCIA01_425_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1011\Brats18_TCIA01_425_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0111\Brats18_TCIA01_425_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1100\Brats18_TCIA01_425_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1010\Brats18_TCIA01_425_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1001\Brats18_TCIA01_425_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0110\Brats18_TCIA01_425_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0101\Brats18_TCIA01_425_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0011\Brats18_TCIA01_425_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1000\Brats18_TCIA01_425_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0100\Brats18_TCIA01_425_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0010\Brats18_TCIA01_425_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0001\Brats18_TCIA01_425_1\predicted_Segm.nii.gz',
    ]

    segs = [
        r'E:\my_files\programmes\python\BRATS2018_normalized\fold4\Brats18_TCIA08_205_1\seg_preprocessed.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1111\Brats18_TCIA08_205_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1110\Brats18_TCIA08_205_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1101\Brats18_TCIA08_205_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1011\Brats18_TCIA08_205_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0111\Brats18_TCIA08_205_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1100\Brats18_TCIA08_205_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1010\Brats18_TCIA08_205_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1001\Brats18_TCIA08_205_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0110\Brats18_TCIA08_205_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0101\Brats18_TCIA08_205_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0011\Brats18_TCIA08_205_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1000\Brats18_TCIA08_205_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0100\Brats18_TCIA08_205_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0010\Brats18_TCIA08_205_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0001\Brats18_TCIA08_205_1\predicted_Segm.nii.gz',
    ]

    segs = [
        r'E:\my_files\programmes\python\BRATS2018_normalized\fold4\Brats18_CBICA_AYI_1\seg_preprocessed.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1111\Brats18_CBICA_AYI_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1110\Brats18_CBICA_AYI_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1101\Brats18_CBICA_AYI_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1011\Brats18_CBICA_AYI_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0111\Brats18_CBICA_AYI_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1100\Brats18_CBICA_AYI_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1010\Brats18_CBICA_AYI_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1001\Brats18_CBICA_AYI_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0110\Brats18_CBICA_AYI_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0101\Brats18_CBICA_AYI_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0011\Brats18_CBICA_AYI_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1000\Brats18_CBICA_AYI_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0100\Brats18_CBICA_AYI_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0010\Brats18_CBICA_AYI_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0001\Brats18_CBICA_AYI_1\predicted_Segm.nii.gz',
    ]

    segs_ref = [
        r'E:\my_files\programmes\python\BRATS2018_normalized\fold4\Brats18_CBICA_AAP_1\seg_preprocessed.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1111\predictions\test_comb0_1111\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1110_ref\predictions\test_comb0_1110\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1101_ref\predictions\test_comb0_1101\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1011_ref\predictions\test_comb0_1011\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0111_ref\predictions\test_comb0_0111\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1100_ref\predictions\test_comb0_1100\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1010_ref\predictions\test_comb0_1010\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1001_ref\predictions\test_comb0_1001\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0110_ref\predictions\test_comb0_0110\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0101_ref\predictions\test_comb0_0101\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0011_ref\predictions\test_comb0_0011\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_1000_ref\predictions\test_comb0_1000\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0100_ref\predictions\test_comb0_0100\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0010_ref\predictions\test_comb0_0010\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_0001_ref\predictions\test_comb0_0001\predictions\Brats18_CBICA_AAP_1\predicted_Segm.nii.gz',
    ]

    # conclude_tests(txt_paths=txts, scenarios=scenarios)
    # plot_comparison_sge_masks(input_paths=segs, slice_num=79, output_path=r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_experiments\experiment5\Brats18_CBICA_AAP_1_seg.png')
    # plot_comparison_sge_masks(input_paths=segs, slice_num=50, output_path=r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_experiments\experiment5\Brats18_TCIA01_425_1.png')
    '''zeros_nii_from(
        r"E:\my_files\programmes\python\mri_gan_output\comb0\test-epoch59\Brats18_CBICA_AAP_1\1110\t1.nii.gz",
        r"E:\my_files\programmes\python\mri_gan_output\comb0"
    )'''
    # conclude_tests(txt_paths=txts_ref, scenarios=scenarios)
    # plot_comparison_sge_masks(input_paths=segs_ref, slice_num=79,
    #                           output_path=r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_experiments\experiment6\Brats18_TCIA01_425_1.png')

    # plot_comparison_sge_masks(input_paths=segs, slice_num=60,
    #                           output_path=r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_experiments\experiment5\Brats18_TCIA08_205_1.png')

    # plot_comparison_sge_masks(input_paths=segs, slice_num=105,
    #                           output_path=r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_experiments\experiment5\Brats18_CBICA_AYI_1.png')

    '''
    compare_volume_with_dir(
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0',
        'seg_preprocessed.nii.gz',
        'predicted_Segm.nii.gz',
        scenarios,
        [0, 1, 2, 4],
        workers=8,
    )
    compare_volume_with_dir(
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0_ref',
        'seg_preprocessed.nii.gz',
        'predicted_Segm.nii.gz',
        scenarios,
        [0, 1, 2, 4],
        workers=8,
    )
    # '''

    scenarios = ['0001', '0010', '0100', '1000',
                 '0011', '0101', '0110', '1001', '1010', '1100',
                 '0111', '1011', '1101', '1110', '1111']
    segs = [
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0001',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0010',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0100',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1000',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0011',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0101',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0110',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1001',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1010',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1100',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\0111',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1011',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1101',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1110',
        r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\1111',
    ]
    pred_nii_name = 'predicted_Segm.nii.gz'
    gt = r'E:\my_files\programmes\python\deepmedic_workplace\deepmedic_outputs\test_comb0\gt'
    gt_nii_name = 'seg.nii.gz'
    for idx, each in enumerate(segs):
        et_avg, ct_avg, wt_avg = compute_dice_with_pred_masks(each, gt, pred_nii_name, gt_nii_name)
        print("%s %f %f %f" % (scenarios[idx], et_avg, ct_avg, wt_avg))

