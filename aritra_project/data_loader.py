# import numpy as np
# import os
# import torch
# from generate_drr import do_full_prprocessing
# import albumentations as A
# from torch.utils.data import DataLoader, Dataset
# import ray
# import psutil


# num_cpus = psutil.cpu_count(logical=False)
# ray.init(num_cpus=num_cpus)

# # dataset paths

# train = "/home/daisylabs/aritra_project/dataset/train"
# val = "/home/daisylabs/aritra_project/dataset/val"
# app = "/home/daisylabs/aritra_project/dataset/app"


# class ImageData(Dataset):
#     def __init__(self, data, phase_coeff):
#         self.root = data
#         self.folder = os.listdir(self.root)
#         self.folder.sort()
#         self.aug = A.Compose([
#             A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=45, interpolation=1, border_mode=4, always_apply=False, p=0.3),
#             A.RandomCrop(220, 220, always_apply=False, p=1.0),
#             A.HorizontalFlip(always_apply=False, p=0.2),
#             A.VerticalFlip(always_apply=False, p=0.2),
#             A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4, always_apply=False, p=0.5),
#             A.RandomBrightness(limit=0.2, always_apply=False, p=0.2),
#             A.RandomContrast(limit=0.2, always_apply=False, p=0.2),
#             A.MedianBlur(blur_limit=5, always_apply=False, p=0.2),
#             A.GaussNoise(var_limit=(10, 50), always_apply=False, p=0.2),
#             A.Resize(256, 256),
#         ])
#         self.phase_coeff = phase_coeff

#     def __len__(self):
#         return (len(self.folder))

#     def __getitem__(self, index):
#         patient_list = os.listdir(os.path.join(self.root, self.folder[index]))
#         patient_list.sort()

#         targets = np.load(os.path.join(self.root, self.folder[index], patient_list[0]))
#         targets = targets.astype('float32')

#         if (self.phase_coeff == 1):
#             targets = np.transpose(targets, (1, 2, 0))

#             transformed = self.aug(image=targets, mask=targets)
#             targets = transformed['mask']
#             targets = (targets - np.min(targets)) * (1.0 / (np.max(targets) - np.min(targets)))
#             targets = np.transpose(targets, (2, 0, 1))

#             targets_ray = ray.put(targets)

#             inputs = ray.get([do_full_prprocessing.remote(targets_ray)])

#             inputs = np.asarray(inputs)

#             inputs[0][1] = np.rot90(inputs[0][1])
#             inputs[0][1] = np.rot90(inputs[0][1])
#             inputs[0][1] = np.rot90(inputs[0][1])


#             inputs = torch.from_numpy(inputs)
#             targets = torch.from_numpy(targets)

#         else:

#             inputs = []

#             inputs_front = np.load(os.path.join(self.root, self.folder[index], patient_list[1]))
#             inputs_lat = np.load(os.path.join(self.root, self.folder[index], patient_list[2]))
#             inputs_top = np.load(os.path.join(self.root, self.folder[index], patient_list[3]))
#             targets = np.load(os.path.join(self.root, self.folder[index], patient_list[0]))

#             inputs_front = inputs_front.astype('float32')
#             inputs_lat = inputs_lat.astype('float32')
#             inputs_top = inputs_top.astype('float32')
#             targets = targets.astype('float32')

#             inputs.append(inputs_front)
#             inputs.append(inputs_lat)
#             inputs.append(inputs_top)

#             inputs = np.array(inputs)

#             inputs = torch.from_numpy(inputs)
#             targets = torch.from_numpy(targets)


#         inputs = inputs.cuda()
#         targets = targets.cuda()

#         return inputs, targets



# def loaders(batch_size, phase):

#     if (phase == 0):
#         dataset = ImageData(train, 1)
#     elif (phase == 1):
#         dataset = ImageData(val, 0)
#     elif (phase == 2):
#         dataset = ImageData(app, 0)

#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=8
#     )

#     return loader


# import numpy as np
# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# from scipy.ndimage import zoom


# # ==============================
# # PATHS
# # ==============================
# train_path = r"D:\Project Data\dataset\train"
# val_path   = r"D:\Project Data\dataset\val"


# # ==============================
# # RESIZE FUNCTIONS
# # ==============================
# def resize_volume(volume, new_size=(256, 256, 256)):
#     factors = (
#         new_size[0] / volume.shape[0],
#         new_size[1] / volume.shape[1],
#         new_size[2] / volume.shape[2],
#     )
#     return zoom(volume, factors, order=1)


# def resize_image(image, new_size=(256, 256)):
#     factors = (
#         new_size[0] / image.shape[0],
#         new_size[1] / image.shape[1],
#     )
#     return zoom(image, factors, order=1)


# # ==============================
# # DATASET
# # ==============================
# class ImageData(Dataset):

#     def __init__(self, root):
#         self.root = root
#         self.patients = sorted(os.listdir(root))

#     def __len__(self):
#         return len(self.patients)

#     def __getitem__(self, index):

#         patient_folder = os.path.join(self.root, self.patients[index])
#         files = sorted(os.listdir(patient_folder))

#         # Expected order:
#         # 0 = CT (512³)
#         # 1 = drrFrontal (512×512)
#         # 2 = drrLateral  (512×512)
#         # 3 = drrTop      (512×512)

#         ct = np.load(os.path.join(patient_folder, files[0])).astype("float32")
#         drr_front = np.load(os.path.join(patient_folder, files[1])).astype("float32")
#         drr_lat   = np.load(os.path.join(patient_folder, files[2])).astype("float32")
#         drr_top   = np.load(os.path.join(patient_folder, files[3])).astype("float32")

#         # =============================
#         # RESIZE 512 → 256
#         # =============================

#         ct = resize_volume(ct, (256, 256, 256))

#         drr_front = resize_image(drr_front, (256, 256))
#         drr_lat   = resize_image(drr_lat, (256, 256))
#         drr_top   = resize_image(drr_top, (256, 256))

#         # Stack 3 DRR views
#         inputs = np.stack([drr_front, drr_lat, drr_top], axis=0)

#         # Convert to tensor
#         inputs = torch.from_numpy(inputs).float()
#         ct = torch.from_numpy(ct).float()

#         return inputs, ct


# # ==============================
# # DATALOADER
# # ==============================
# def loaders(batch_size, phase):

#     if phase == 0:
#         dataset = ImageData(train_path)
#     elif phase == 1:
#         dataset = ImageData(val_path)
#     else:
#         raise ValueError("Invalid phase")

#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=(phase == 0),
#         num_workers=0,   # REQUIRED on Windows
#         pin_memory=False
#     )

#     return loader

import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom

# ==============================
# PATHS
# ==============================
train_path = r"D:\Project Data\dataset\train"
val_path   = r"D:\Project Data\dataset\val"


# ==============================
# RESIZE FUNCTIONS
# ==============================
def resize_volume(volume, new_size=(256, 256, 256)):
    factors = (
        new_size[0] / volume.shape[0],
        new_size[1] / volume.shape[1],
        new_size[2] / volume.shape[2],
    )
    return zoom(volume, factors, order=1)


def resize_image(image, new_size=(256, 256)):
    factors = (
        new_size[0] / image.shape[0],
        new_size[1] / image.shape[1],
    )
    return zoom(image, factors, order=1)


# ==============================
# DATASET
# ==============================
class ImageData(Dataset):

    def __init__(self, root):
        self.root = root
        self.patients = sorted(os.listdir(root))

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, index):

        patient_folder = os.path.join(self.root, self.patients[index])
        files = sorted(os.listdir(patient_folder))

        ct = np.load(os.path.join(patient_folder, files[0])).astype(np.float32)
        drr_front = np.load(os.path.join(patient_folder, files[1])).astype(np.float32)
        drr_lat   = np.load(os.path.join(patient_folder, files[2])).astype(np.float32)
        drr_top   = np.load(os.path.join(patient_folder, files[3])).astype(np.float32)

        # =============================
        # RESIZE (CPU SIDE)
        # =============================
        ct = resize_volume(ct, (256, 256, 256))

        drr_front = resize_image(drr_front, (256, 256))
        drr_lat   = resize_image(drr_lat, (256, 256))
        drr_top   = resize_image(drr_top, (256, 256))

        # inputs = np.stack([drr_front, drr_lat, drr_top], axis=0) for multi View Baseline
        inputs = np.stack([drr_front], axis=0) # for 1 - view baseline
        

        # Convert to tensor (no extra copy)
        inputs = torch.from_numpy(inputs)
        ct = torch.from_numpy(ct)

        return inputs, ct


# ==============================
# DATALOADER
# ==============================
def loaders(batch_size, phase):

    if phase == 0:
        dataset = ImageData(train_path)
    elif phase == 1:
        dataset = ImageData(val_path)
    else:
        raise ValueError("Invalid phase")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(phase == 0),
        num_workers=4,          #  Use CPU cores
        pin_memory=True,        # aster GPU transfer
        persistent_workers=True # Avoid worker restart
    )

    return loader