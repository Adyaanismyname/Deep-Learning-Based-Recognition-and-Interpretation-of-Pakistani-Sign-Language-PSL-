import os

directory = "Dataset_Augmented"

for dir in os.listdir(directory):
    if dir == ".DS_Store":
        continue
    print(f"{dir} : {len(os.listdir(os.path.join(directory,dir)))}")