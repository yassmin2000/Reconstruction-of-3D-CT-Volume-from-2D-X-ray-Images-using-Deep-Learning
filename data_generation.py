import numpy as np # linear algebra
import pydicom
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from configparser import ConfigParser
import warnings
import pylidc as pl
from numba import jit
import ray
from skimage import io
import psutil
from scipy.ndimage import zoom
import cv2
warnings.filterwarnings(action='ignore')
import os
from pydicom.errors import InvalidDicomError


def is_dicom_file(fp: str) -> bool:
    # LIDC files sometimes have no extension; also can be .dcm
    name = os.path.basename(fp).lower()
    return name.endswith(".dcm") or "." not in name

def find_dicom_series_folder(patient_dir: str, min_files: int = 50) -> str | None:
    """
    Walk inside patient_dir and return a folder that looks like a CT series:
    - contains many DICOM files (>= min_files)
    """
    best = None
    best_count = 0

    for root, _, files in os.walk(patient_dir):
        dicom_files = [f for f in files if is_dicom_file(os.path.join(root, f))]
        if len(dicom_files) > best_count and len(dicom_files) >= min_files:
            best = root
            best_count = len(dicom_files)

    return best
def plot_3d(image, threshold=-300):
	# Position the scan upright,
	# so the head of the patient would be at the top facing the camera
	p = image.transpose(2, 1, 0)

	verts, faces, normals, values = measure.marching_cubes_lewiner(p, threshold)

	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(111, projection='3d')

	# Fancy indexing: `verts[faces]` to generate a collection of triangles
	mesh = Poly3DCollection(verts[faces], alpha=0.70)
	face_color = [0.45, 0.45, 0.75]
	mesh.set_facecolor(face_color)
	ax.add_collection3d(mesh)

	ax.set_xlim(0, p.shape[0])
	ax.set_ylim(0, p.shape[1])
	ax.set_zlim(0, p.shape[2])

	plt.show()

# Load the scans in given folder path
def load_scan(path):
    slices = []
    for fname in os.listdir(path):
        fp = os.path.join(path, fname)
        if not os.path.isfile(fp):
            continue
        try:
            ds = pydicom.dcmread(fp, force=True)
        except (InvalidDicomError, Exception):
            continue

        # Keep only CT image slices (skip SR, RTSTRUCT, etc.)
        if getattr(ds, "Modality", None) != "CT":
            continue
        if not hasattr(ds, "PixelData"):
            continue
        if not hasattr(ds, "ImagePositionPatient"):
            continue

        slices.append(ds)

    if len(slices) < 2:
        raise RuntimeError(f"Not enough CT slices found in: {path}")

    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # Slice thickness
    try:
        slice_thickness = abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except Exception:
        slice_thickness = abs(getattr(slices[0], "SliceLocation", 0) - getattr(slices[1], "SliceLocation", 0))

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def convert_dcm_to_npy(images):
	volume = np.stack(
		[
			x.pixel_array * x.RescaleSlope + x.RescaleIntercept
			for x in images
		],
		axis=-1,
	).astype(np.int16)
	return volume


def get_pixels_hu(slices):
	image = np.stack([s.pixel_array for s in slices])
	# Convert to int16 (from sometimes int16),
	# should be possible as values should always be low enough (<32k)
	image = image.astype(np.int16)

	# Set outside-of-scan pixels to 0
	# The intercept is usually -1024, so air is approximately 0
	image[image == -2000] = 0

	# Convert to Hounsfield units (HU)
	for slice_number in range(len(slices)):

		intercept = slices[slice_number].RescaleIntercept
		slope = slices[slice_number].RescaleSlope

		if slope != 1:
			image[slice_number] = slope * image[slice_number].astype(np.float64)
			image[slice_number] = image[slice_number].astype(np.int16)

		image[slice_number] += np.int16(intercept)

	return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[1, 1, 1]):
	# Determine current pixel spacing
	spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)

	resize_factor = spacing / new_spacing
	new_real_shape = image.shape * resize_factor
	new_shape = np.round(new_real_shape)
	real_resize_factor = new_shape / image.shape
	new_spacing = spacing / real_resize_factor

	image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

	return image, new_spacing

@jit(nopython=True, parallel=True)
def generate_drr_from_ct(ct_scan, direction='frontal'):
	input_shape = ct_scan.shape
	if direction == 'lateral':
		ct_scan = np.transpose(ct_scan, axes=(0, 2, 1))
		input_shape = ct_scan.shape
	elif direction == "top":
		ct_scan = np.transpose(ct_scan, axes=(1, 0, 2))
		input_shape = ct_scan.shape

	drr_out = np.zeros((input_shape[0], input_shape[2]), dtype=np.float32)
	for x in range(input_shape[0]):
		for z in range(input_shape[2]):
			u_av = 0.0
			for y in range(input_shape[1]):
				u_av += 0.2 * (ct_scan[x, y, z] + 1000) / (input_shape[1] * 1000)
			drr_out[x, z] = np.exp(0.02 + u_av)
	return drr_out

@ray.remote
def do_full_prprocessing(patients, output_folder, pat_idxs):
	out_meta = []
	for i in pat_idxs:
		scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patients[i]).first()
		dcm_slices = scan.load_all_dicom_images()
		patient_pixels = get_pixels_hu(dcm_slices)
		pix_resampled, spacing = resample(patient_pixels, dcm_slices, [1, 1, 1])
		if not os.path.isdir(os.path.join(output_folder, patients[i])):
			os.makedirs(os.path.join(output_folder, patients[i]))


		drr_front = generate_drr_from_ct(pix_resampled, direction='frontal')
		drr_lat = generate_drr_from_ct(pix_resampled, direction='lateral')
		drr_top = generate_drr_from_ct(pix_resampled, direction='top')

		pix_resampled = np.transpose(pix_resampled, axes=(1, 0, 2))
		org_shape = pix_resampled.shape
		pix_resampled = zoom(pix_resampled, (512 / org_shape[0], 512 / org_shape[1], 512 / org_shape[2]))
		pix_resampled = (pix_resampled - np.min(pix_resampled)) * (1.0 / (np.max(pix_resampled) - np.min(pix_resampled)))
		np.save(os.path.join(output_folder, patients[i], f"{patients[i]}.npy"), pix_resampled)

		drr_front = cv2.resize(drr_front, (512, 512), interpolation=cv2.INTER_LINEAR)
		drr_front = (drr_front - np.min(drr_front)) * (1.0 / (np.max(drr_front) - np.min(drr_front)))
		np.save(os.path.join(output_folder, patients[i], f"{patients[i]}_drrFrontal.npy"), drr_front)

		drr_lat = cv2.resize(drr_lat, (512, 512), interpolation=cv2.INTER_LINEAR)
		drr_lat = (drr_lat - np.min(drr_lat)) * (1.0 / (np.max(drr_lat) - np.min(drr_lat)))
		np.save(os.path.join(output_folder, patients[i], f"{patients[i]}_drrLateral.npy"), drr_lat)

		drr_top = cv2.resize(drr_top, (512, 512), interpolation=cv2.INTER_LINEAR)
		drr_top = (drr_top - np.min(drr_top)) * (1.0 / (np.max(drr_top) - np.min(drr_top)))
		np.save(os.path.join(output_folder, patients[i], f"{patients[i]}_drrTop.npy"), drr_top)

		out_meta.append((i, spacing))
	return out_meta


# Read the configuration file generated from config_file_create.py
parser = ConfigParser()
parser.read('./lidc.conf')

# Some constants
input_folder = r"D:/UofA/Courses/ECE 740/Reconstruction-of-3D-CT-Volume-from-2D-X-ray-Images-using-Deep-Learning/LIDC_dataset/LIDC-IDRI"
output_folder = r"D:/UofA/Courses/ECE 740/Reconstruction-of-3D-CT-Volume-from-2D-X-ray-Images-using-Deep-Learning/DRRs"

patients = sorted([p for p in os.listdir(input_folder) if p.startswith("LIDC-IDRI-")])
os.makedirs(output_folder, exist_ok=True)

for pid in patients:
    patient_dir = os.path.join(input_folder, pid)
    series_dir = find_dicom_series_folder(patient_dir, min_files=50)
    if series_dir is None:
        print(f"[SKIP] {pid}: no series folder found")
        continue

    try:
        dcm_slices = load_scan(series_dir)                 # <-- now robust
        patient_pixels = get_pixels_hu(dcm_slices)
        pix_resampled, spacing = resample(patient_pixels, dcm_slices, [1, 1, 1])

        out_dir = os.path.join(output_folder, pid)
        os.makedirs(out_dir, exist_ok=True)

        drr_front = generate_drr_from_ct(pix_resampled, direction='frontal')
        drr_lat   = generate_drr_from_ct(pix_resampled, direction='lateral')
        drr_top   = generate_drr_from_ct(pix_resampled, direction='top')
        pix_resampled = np.transpose(pix_resampled, axes=(1, 0, 2))
        org_shape = pix_resampled.shape
        pix_resampled = zoom(pix_resampled, (512 / org_shape[0], 512 / org_shape[1], 512 / org_shape[2]))
        pix_resampled = (pix_resampled - np.min(pix_resampled)) / (np.max(pix_resampled) - np.min(pix_resampled) + 1e-8)
        np.save(os.path.join(out_dir, f"{pid}.npy"), pix_resampled)
        def save_drr(drr, name):
                  drr = cv2.resize(drr, (512, 512), interpolation=cv2.INTER_LINEAR)
                  drr = (drr - np.min(drr)) / (np.max(drr) - np.min(drr) + 1e-8)
                  np.save(os.path.join(out_dir, name), drr)
        save_drr(drr_front, f"{pid}_drrFrontal.npy")
        save_drr(drr_lat,   f"{pid}_drrLateral.npy")
        save_drr(drr_top,   f"{pid}_drrTop.npy")
        print(f"[OK] {pid}: {len(dcm_slices)} slices from {series_dir}")

    except Exception as e:
        print(f"[FAIL] {pid}: {e}")

