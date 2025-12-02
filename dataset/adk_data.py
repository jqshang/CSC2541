import numpy as np
import MDAnalysis as mda
from MDAnalysisData import datasets
from MDAnalysis.analysis.dihedrals import Ramachandran


def generate_adk_data():

    adk = datasets.fetch_adk_equilibrium()
    u = mda.Universe(adk.topology, adk.trajectory)
    ca = u.select_atoms("protein and name CA")

    T = len(u.trajectory)
    N = len(ca)
    pos = np.zeros((T, N, 3))

    for t, ts in enumerate(u.trajectory):
        pos[t] = ca.positions.copy()

    rama = Ramachandran(ca)
    rama.run()
    angles = rama.angles  # shape: (T, 212, 2)

    rama_residues = list(rama.ag3.residues)
    rama_resids = np.array([res.resid for res in rama_residues])

    ca_resids = np.array([res.resid for res in ca.residues])

    resid_to_index = {resid: i for i, resid in enumerate(ca_resids)}
    valid_residue_indices = np.array([resid_to_index[r] for r in rama_resids])

    positions = pos[:, valid_residue_indices]  # (T, 212, 3)

    num_residues = len(rama_residues)  # 212
    amino_acids = [f"A{i+1}"
                   for i in range(num_residues)]  # ["A1", ..., "A212"]

    label_to_residue = {
        label: {
            "index": i,
            "resid": res.resid,
            "resname": res.resname
        }
        for i, (label, res) in enumerate(zip(amino_acids, rama_residues))
    }

    resid_to_label = {
        info["resid"]: label
        for label, info in label_to_residue.items()
    }

    return positions, angles, amino_acids, amino_acids
