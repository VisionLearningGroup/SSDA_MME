import subprocess
import os
p_path = "/home/grad3/keisaito/project/da/semisupervised/data/txt/office"
txts = os.listdir(p_path)
for txt in txts:
    txt_path = os.path.join(p_path, txt)
    lines = open(txt_path, "r").readlines()
    new_file = open(txt_path, "w")
    for line in lines:
        file_path = line.split(" ")[0].split("/")[-1]
        dir_path = line.split(" ")[0].split("/")[-2]
        d_path = line.split(" ")[0].split("/")[-4]
        class_n = line.split(" ")[1]
        new_file.write(os.path.join(d_path, "images", dir_path, file_path) + " " + class_n)


