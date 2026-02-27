import os
import numpy as np
import matplotlib.pyplot as plt

def check_patient_folder(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
    if len(files) < 4:
        return False, f"Expected >=4 .npy files, found {len(files)}: {files}"

    # Try to locate CT + DRRs by name
    ct_path = None
    drr_paths = {}
    for f in files:
        low = f.lower()
        fp = os.path.join(folder, f)
        if low.endswith("_drrfrontal.npy"):
            drr_paths["frontal"] = fp
        elif low.endswith("_drrlateral.npy"):
            drr_paths["lateral"] = fp
        elif low.endswith("_drrtop.npy"):
            drr_paths["top"] = fp
        else:
            # assume the non-drr is CT
            if ct_path is None:
                ct_path = fp

    if ct_path is None or len(drr_paths) != 3:
        return False, f"Could not identify CT + 3 DRRs. CT={ct_path}, DRRs={list(drr_paths.keys())}"

    ct = np.load(ct_path)
    drr_f = np.load(drr_paths["frontal"])
    drr_l = np.load(drr_paths["lateral"])
    drr_t = np.load(drr_paths["top"])

    # --- Shape checks ---
    if ct.ndim != 3:
        return False, f"CT must be 3D but got shape {ct.shape}"
    for name, drr in [("frontal", drr_f), ("lateral", drr_l), ("top", drr_t)]:
        if drr.ndim != 2:
            return False, f"DRR {name} must be 2D but got shape {drr.shape}"

    H, W = drr_f.shape
    if drr_l.shape != (H, W) or drr_t.shape != (H, W):
        return False, f"DRR shapes mismatch: frontal={drr_f.shape}, lateral={drr_l.shape}, top={drr_t.shape}"

    # CT should be cubic in your pipeline, and match DRR size in x/y
    if ct.shape[0] != ct.shape[1] or ct.shape[1] != ct.shape[2]:
        print(f"[WARN] CT not cubic: {ct.shape} (might still be ok if your training expects this)")
    if (ct.shape[0] != H) and (ct.shape[1] != H) and (ct.shape[2] != H):
        print(f"[WARN] CT size {ct.shape} doesn't obviously match DRR size {(H,W)}")

    # --- Value checks ---
    def stats(x):
        return float(np.min(x)), float(np.max(x)), float(np.mean(x)), float(np.std(x))

    for name, arr in [("CT", ct), ("DRR frontal", drr_f), ("DRR lateral", drr_l), ("DRR top", drr_t)]:
        if not np.isfinite(arr).all():
            return False, f"{name} has NaN/Inf"
        mn, mx, mu, sd = stats(arr)
        if sd < 1e-6:
            return False, f"{name} looks constant (std~0): min={mn} max={mx} mean={mu}"
        # if normalized, should be near [0,1]
        if mn < -0.05 or mx > 1.05:
            print(f"[WARN] {name} not in [0,1] range: min={mn:.3f} max={mx:.3f}")

    # --- Geometric sanity (quick correlation with simple projections) ---
    # Simple projections (mean along axis) - choose an axis that matches DRR size
    proj0 = ct.mean(axis=0)  # (H,W) if ct is (D,H,W)
    proj1 = ct.mean(axis=1)
    proj2 = ct.mean(axis=2)

    # pick the projection that matches DRR size best
    projs = [proj0, proj1, proj2]
    proj = min(projs, key=lambda p: abs(p.shape[0] - H) + abs(p.shape[1] - W))

    # normalize projection for display
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-8)

    # show quick visual comparison
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    ax[0].imshow(proj, cmap="gray"); ax[0].set_title("CT mean projection")
    ax[1].imshow(drr_f, cmap="gray"); ax[1].set_title("DRR frontal")
    ax[2].imshow(drr_l, cmap="gray"); ax[2].set_title("DRR lateral")
    ax[3].imshow(drr_t, cmap="gray"); ax[3].set_title("DRR top")
    for a in ax: a.axis("off")
    plt.tight_layout()
    plt.show()

    return True, f"OK: CT={ct.shape}, DRRs={drr_f.shape}"

def validate_root(root, max_patients=3):
    patients = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    patients = patients[:max_patients]
    for p in patients:
        folder = os.path.join(root, p)
        ok, msg = check_patient_folder(folder)
        print(f"{'[PASS]' if ok else '[FAIL]'} {p}: {msg}")
        if not ok:
            break

if __name__ == "__main__":
    # change this to your DRRs output folder
    root = r"D:\UofA\Courses\ECE 740\Reconstruction-of-3D-CT-Volume-from-2D-X-ray-Images-using-Deep-Learning\DRRs"
    validate_root(root, max_patients=3)