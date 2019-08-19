import subprocess


domains =["amazon", "dslr", "webcam"]
p1 = "/research/masaito/office/source_images_%s.txt"
p2 = "/research/masaito/office/split_iccv/labeled_target_images_%s_3.txt"
p3 = "/research/masaito/office/split_iccv/unsupervised_target_images_%s_3.txt"
p4 = "/research/masaito/office/split_iccv/validation_target_images_%s_3.txt"
paths = [p1, p2, p3, p4]
for dom in domains:
    for i, p in enumerate(paths):
        txt = p % dom
        if i == 0:
            cmd = "cp " + txt + " ../data/txt/labeled_source_images_%s.txt" % dom
        elif i == 2:
            cmd = "cp " + txt + " ../data/txt/unlabeled_target_images_%s.txt" % dom
        else:
            cmd = "cp " + txt + " ../data/txt/."

        print(cmd)
        subprocess.call(cmd, shell=True)
