import os

cancer_type = "benign"
subtype = "adenosis"
base_path = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/OxML/BreaKHis_v1/BreaKHis_v1/histology_slides/breast"
image_names = os.listdir(base_path)

labels = []
file_names = []

magnifications = [200]
for cancer_type in ["benign", "malignant"]:
    cancer_path = f"{base_path}/{cancer_type}/SOB"
    for subtype in os.listdir(cancer_path):
        for slide in os.listdir(f"{cancer_path}/{subtype}"):
            for magnif in magnifications:
                slide_path = f"{cancer_path}/{subtype}/{slide}/{magnif}X"
                if not os.path.isdir(slide_path):
                    continue
                image_names = [f"{slide_path}/{i}" for i in os.listdir(slide_path)]
                file_names.append(image_names)
                labels.append([cancer_type]*len(image_names))

file_names = [item for sublist in file_names for item in sublist]
labels = [item for sublist in labels for item in sublist]

with open("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/OxML/BreaKHis_v1/img_paths.txt", 'w') as csvfile:
    for file_name in file_names:
        csvfile.write("%s\n" % (file_name))

with open("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/OxML/BreaKHis_v1/labels.txt", 'w') as csvfile:
    for label in labels:
        csvfile.write("%s\n" % (label))
