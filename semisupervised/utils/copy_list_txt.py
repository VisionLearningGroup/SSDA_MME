import subprocess


domains =["real", "clipart", "painting", "sketch"]
p1 = "/research/masaito/multisource_data/few_shot_DA_data/split_iccv/labeled_source_images_%s.txt"
p2 = "/research/masaito/multisource_data/few_shot_DA_data/split_iccv/labeled_target_images_%s_3.txt"
p3 = "/research/masaito/multisource_data/few_shot_DA_data/split_iccv/unlabeled_target_images_%s_3.txt"
p4 = "/research/masaito/multisource_data/few_shot_DA_data/split_iccv/validation_target_images_%s_3.txt"
paths = [p1, p2, p3, p4]
for dom in domains:
    for p in paths:
        txt = p % dom
        cmd = "cp " + txt + " ../data/txt/."
        print(cmd)
        subprocess.call(cmd, shell=True)
