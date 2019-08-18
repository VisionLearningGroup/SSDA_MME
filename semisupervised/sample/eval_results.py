import sys
import numpy as np


def return_label_list(file):
    tmp = open(file, "r")
    lines = tmp.readlines()
    labels = [int(line.strip().split(" ")[1]) for line in lines]
    file_name = [line.strip().split(" ")[0] for line in lines]
    return labels, file_name


def return_label2cat(file):
    tmp = open(file, "r")
    lines = tmp.readlines()
    dict = {}
    for line in lines:
        dict[int(line.strip().split(" ")[0])] = line.strip().split(" ")[1]
    return dict


def check_file_name(file1, file2):
    for i, fil_1 in enumerate(file1):
        fil_2 = file2[i]
        if fil_1 != fil_2:
            return False
    return True


def acc_perclass(gt, submit, label2cat_file, output="sample.txt"):
    gt_labels, file_gt = return_label_list(gt)
    submit_labels, file_sb = return_label_list(submit)
    label2cat = return_label2cat(label2cat_file)
    try:
        assert len(gt_labels) == len(submit_labels)
    except:
        print('Number of submitted files and GT is different!')
    try:
        assert check_file_name(file_gt, file_sb)
    except:
        print('Submitted files do not correpond to GT files!')
    result_np = np.zeros((max(gt_labels) + 1, max(gt_labels) + 1))
    for i, gt in enumerate(gt_labels):
        submit_l = submit_labels[i]
        result_np[gt, submit_l] += 1
    mean_acc_all = np.sum(np.diag(result_np)) / len(submit_labels) * 100.0
    acc_for_classes = np.diag(result_np) / np.sum(result_np, axis=1)
    mean_acc_classes = np.mean(acc_for_classes) * 100.0
    print("ACC All %f ACC Averaged over Classes %f" % (mean_acc_all, mean_acc_classes))
    with open(output, "w") as out_per_class:
        for label, acc in enumerate(acc_for_classes):
            cat_name = label2cat[label]
            out_per_class.write("%s: %f\n" % (cat_name, acc))


gt_file = sys.argv[1]
submit_file = sys.argv[2]
label2category = sys.argv[3]
acc_perclass(gt_file, submit_file, label2category)
