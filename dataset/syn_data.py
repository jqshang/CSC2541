import numpy as np


def generate_synthetic_protein_data(num_acids, num_steps, pairs, seed=42):
    np.random.seed(seed)
    amino_acids = [f"A{i}" for i in range(1, num_acids + 1)]
    idx = {a: i for i, a in enumerate(amino_acids)}

    pos = np.zeros((num_steps, num_acids, 3))
    ang = np.zeros((num_steps, num_acids, 2))

    pos[0] = np.random.uniform(-1, 1, size=(num_acids, 3))
    ang[0] = np.random.uniform(-180, 180, size=(num_acids, 2))

    coupling_strength = 0.1
    noise_level = 0.01

    for t in range(num_steps - 1):
        pos[t +
            1] = pos[t] + np.random.normal(0, noise_level, size=(num_acids, 3))
        ang[t +
            1] = ang[t] + np.random.normal(0, noise_level, size=(num_acids, 2))

        for Ai, Aj in pairs:
            i = idx[Ai]
            j = idx[Aj]

            Ai_pos = pos[t, i]
            Ai_ang = ang[t, i]

            pos[t + 1, j] += np.sin(Ai_pos) * coupling_strength
            ang[t + 1, j] -= Ai_ang * coupling_strength

        ang[t + 1] = ((ang[t + 1] + 180) % 360) - 180

    return pos, ang, amino_acids
