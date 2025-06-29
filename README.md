# Calculation-of-Native-Contacts-with-Enlarged-VDW-Radii-using-GPU-Acceleration
This repository provides a GPU-accelerated algorithm for computing native contacts across all frames of a molecular dynamics trajectory. The calculation is based on overlap of enlarged van der Waals (VDW) radii, statistically derived for each amino acid, as described in:
> **Statistical radii associated with amino acids to determine the contact map: fixing the structure of a type I cohesin domain in the *Clostridium thermocellum* cellulosome**  
> Mateusz Chwastyk, Adolfo Poma Bernaola, and Marek Cieplak  
> *Physical Biology*, Volume 12, Number 4 (2015)  
> DOI: [10.1088/1478-3975/12/4/046002](https://doi.org/10.1088/1478-3975/12/4/046002)

### Features
- GPU-accelerated for fast trajectory processing
- Analyzes all frames from a given trajectory
- Computes contact frequencies across all frames
- Input: PDB structure + trajectory
- Output: `output.txt` with native contact pairs and frequency
   
### Requirements
- Python ≥ 3.7
- NumPy
- MDAnalysis
- CuPy v12.x *(depending on cuda version)*

### Installation

```bash
git clone (this_repo)
cd (repo)
pip install -r requirements.txt
pip install -e .
```

### Usage
```bash
evdw_nc --s structure.pdb --f trajectory.xtc --o output.txt
```
--s: Path to the input PDB file containing the protein or complex structure (residues must be numbered sequentially).
--f: Path to the trajectory file (e.g., XTC format).
--o: (Optional) Output filename where native contact frequencies will be saved.
