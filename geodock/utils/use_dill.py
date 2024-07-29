import dill
import numpy as np

def load_dill(file):
    f = open(file, 'rb')
    df = dill.load(f)
    return df

def get_seqs(file):
    df = load_dill(file)
    seq1 = df[7]['r_b']
    seq2 = df[7]['l_b']
    return seq1, seq2

def get_coords(file):
    rec_coords = get_rec_coords(file)
    lig_coords = get_lig_coords(file)

    rec = np.stack([rec_coords['N'], rec_coords['CA'], rec_coords['C']], axis=1)
    lig = np.stack([lig_coords['N'], lig_coords['CA'], lig_coords['C']], axis=1)
    return rec, lig

def get_lig_coords(file):
    df = load_dill(file)
    coords = df[1]
    n_coords = np.array(coords[coords['atom_name'] == 'N'][['x', 'y', 'z']])
    ca_coords = np.array(coords[coords['atom_name'] == 'CA'][['x', 'y', 'z']])
    c_coords = np.array(coords[coords['atom_name'] == 'C'][['x', 'y', 'z']])
    o_coords = np.array(coords[coords['atom_name'] == 'O'][['x', 'y', 'z']])

    atom_coords = {}
    atom_coords['N'] = n_coords
    atom_coords['CA'] = ca_coords
    atom_coords['C'] = c_coords
    atom_coords['O'] = o_coords

    return atom_coords

def get_rec_coords(file):
    df = load_dill(file)
    coords = df[2]
    n_coords = np.array(coords[coords['atom_name'] == 'N'][['x', 'y', 'z']])
    ca_coords = np.array(coords[coords['atom_name'] == 'CA'][['x', 'y', 'z']])
    c_coords = np.array(coords[coords['atom_name'] == 'C'][['x', 'y', 'z']])
    o_coords = np.array(coords[coords['atom_name'] == 'O'][['x', 'y', 'z']])

    atom_coords = {}
    atom_coords['N'] = n_coords
    atom_coords['CA'] = ca_coords
    atom_coords['C'] = c_coords
    atom_coords['O'] = o_coords

    return atom_coords

def get_backbone(df):
    n_coords = np.array(df[df['atom_name'] == 'N'][['x', 'y', 'z']])
    ca_coords = np.array(df[df['atom_name'] == 'CA'][['x', 'y', 'z']])
    c_coords = np.array(df[df['atom_name'] == 'C'][['x', 'y', 'z']])
    residues = df[df['atom_name'] == 'CA']['resname']
    res = [x for x in residues.tolist()]
    pos = np.stack([n_coords, ca_coords, c_coords], axis=1)
    assert len(n_coords) == len(ca_coords) == len(c_coords) == len(res)

    backbone = {}
    backbone['res'] = res
    backbone['pos'] = pos
    return backbone

def get_ca_coords(df):
    residues = df[df['atom_name'] == 'CA']['resname']
    res = [x for x in residues.tolist()]
    pos = np.array(df[df['atom_name'] == 'CA'][['x', 'y', 'z']])
    assert len(pos) == len(res)

    data = {}
    data['res'] = res
    data['pos'] = pos
    return data

def get_data(file):
    df = load_dill(file)
    rec = get_ca_coords(df[1])
    lig = get_ca_coords(df[2])
    return {'receptor' :rec, 'ligand': lig}




#===test===
if __name__ == '__main__':
    file_name = '/home/lchu11/scr4_jgray21/lchu11/Docking-dev/data/dips/raw/ke/1keu.pdb1_0.dill'
    data = get_data(file_name)
    print(data)