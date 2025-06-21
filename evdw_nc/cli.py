import mdtraj as md
import numpy as np
import sys
import time
import argparse

# GPU acceleration attempt using CuPy
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    print("[WARN] CuPy not installed, falling back to CPU (NumPy only).")
    GPU_AVAILABLE = False

# === CONFIGURATION ===
SEPARACION_MINIMA = 3  # |j - i| >= 3
BLOCK_SIZE = 512       # Block size for GPU operations

# === VDW RADII (in Å) ===
VDW_RADII = {
    "GLY":3.15, "ALA":3.35, "SER":3.30, "ASP":3.50, "THR":3.60,
    "ASN":3.65, "GLU":3.65, "LYS":3.65, "CYS":3.70, "PRO":3.70,
    "GLN":3.90, "ARG":3.95, "VAL":4.00, "HIS":4.00, "HID":4.00,
    "HIE":4.00, "HIP":4.00, "ILE":4.50, "MET":4.50, "TYR":4.50,
    "LEU":4.60, "PHE":4.60, "TRP":4.70
}

def process_frame_gpu(frame, vdw_vec, freq_mat, inv_nframes):
    n_res = frame.shape[0]
    for i0 in range(0, n_res, BLOCK_SIZE):
        i1 = min(n_res, i0 + BLOCK_SIZE)
        sub_i = frame[i0:i1]
        for j0 in range(i0 + SEPARACION_MINIMA, n_res, BLOCK_SIZE):
            j1 = min(n_res, j0 + BLOCK_SIZE)
            diffs = sub_i[:, None, :] - frame[j0:j1][None, :, :]
            d2 = cp.sum(diffs**2, axis=2)
            thresh = (vdw_vec[i0:i1, None] + vdw_vec[None, j0:j1])**2
            mask = d2 <= thresh
            freq_mat[i0:i1, j0:j1] += mask.astype(freq_mat.dtype) * inv_nframes

def main():
    parser = argparse.ArgumentParser(description='Calculate VdW contact frequencies using MDTraj.')
    parser.add_argument('--s', required=True, help='Structure file (i.e. pdb, parm7)')
    parser.add_argument('--f', required=True, help='Trajectory file (i.e xtc, dcd, nc)')
    parser.add_argument('--o', default='output.txt', help='Output file (default: output.txt)')
    args = parser.parse_args()

    pdb = args.s
    xtc = args.f
    output_file = args.o

    print("[INFO] Loading trajectory...")
    t0 = time.time()
    traj = md.load(xtc, top=pdb)
    load_time = time.time() - t0
    print(f"[INFO] Trajectory loaded in {load_time:.1f} s")

    top = traj.topology

    # --- Select atoms: prefer BB, fallback to CA ---
    atom_map = {}
    for atom in top.atoms:
        res_id = atom.residue.index
        if res_id not in atom_map and atom.name in ('BB', 'CA'):
            atom_map[res_id] = atom.index

    if not atom_map:
        sys.exit("[ERROR] No BB or CA atoms found.")

    res_ids = sorted(atom_map)
    res_ids = [r + 1 for r in res_ids]
    atom_idx = [atom_map[r - 1] for r in res_ids]
    coords = traj.xyz[:, atom_idx, :]  # in nm

    n_frames, n_res, _ = coords.shape
    print(f"[INFO] Frames={n_frames}, Residues={n_res}, GPU={GPU_AVAILABLE}")

    # Convert VDW radii to nm
    vdw_vals = np.array([VDW_RADII.get(top.residue(r-1).name, 0) * 0.1 for r in res_ids])

    if GPU_AVAILABLE:
        coords = cp.asarray(coords)
        vdw_vals = cp.asarray(vdw_vals)
        freq_mat = cp.zeros((n_res, n_res), dtype=cp.float32)
        inv_nframes = cp.float32(1.0 / n_frames)
    else:
        freq_mat = np.zeros((n_res, n_res), dtype=np.float32)
        inv_nframes = np.float32(1.0 / n_frames)

    # --- Compute VDW contact frequencies ---
    print("[INFO] Computing contact frequencies frame by frame...")
    t1 = time.time()
    for f in range(n_frames):
        frame = coords[f]
        if GPU_AVAILABLE:
            process_frame_gpu(frame, vdw_vals, freq_mat, inv_nframes)
        else:
            for i in range(n_res - SEPARACION_MINIMA):
                for j in range(i + SEPARACION_MINIMA, n_res):
                    if np.linalg.norm(frame[i] - frame[j]) <= (vdw_vals[i] + vdw_vals[j]):
                        freq_mat[i, j] += inv_nframes
        if (f + 1) % max(1, n_frames // 100) == 0 or f == n_frames - 1:
            pct = (f + 1) / n_frames * 100
            sys.stdout.write(f"\r  Processing frame {f+1}/{n_frames} ({pct:.1f}%)")
            sys.stdout.flush()
    sys.stdout.write("\n")
    frames_time = time.time() - t1
    print(f"[INFO] Frequency calculation completed in {frames_time:.1f} s")

    if GPU_AVAILABLE:
        freq_mat = freq_mat.get()

    # --- Write results ---
    print("[INFO] Writing frequencies to file...")
    t2 = time.time()
    i_idx, j_idx = np.nonzero(np.triu(freq_mat, k=SEPARACION_MINIMA))
    total = len(i_idx)
    with open(output_file, "w") as f:
        for idx, (ii, jj) in enumerate(zip(i_idx, j_idx), start=1):
            f.write(f"{res_ids[ii]} {res_ids[jj]} {freq_mat[ii, jj]:.2f}\n")
            if idx % max(1, total // 100) == 0 or idx == total:
                pct = idx / total * 100
                sys.stdout.write(f"\r  Writing {idx}/{total} ({pct:.1f}%)")
                sys.stdout.flush()
    sys.stdout.write("\n")
    write_time = time.time() - t2
    total_time = time.time() - t0
    print(f"[INFO] Writing completed in {write_time:.1f} s")
    print(f"[INFO] Total process time: {total_time:.1f} s")

if __name__ == "__main__":
    main()

