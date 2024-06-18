import pandas as pd
import pickle

# taken from the TANKBind GitHub repository
# Copyright (c) 2022 Wei Lu, Galixir Technologies
def read_pdbbind_data(fileName):
    with open(fileName) as f:
        a = f.readlines()
    info = []
    for line in a:
        if line[0] == '#':
            continue
        lines, ligand = line.split('//')
        pdb, resolution, year, affinity, raw = lines.strip().split('  ')
        ligand = ligand.strip().split('(')[1].split(')')[0]
        info.append([pdb, resolution, year, affinity, raw, ligand])
    info = pd.DataFrame(info, columns=['pdb', 'resolution', 'year', 'affinity', 'raw', 'ligand'])
    info.year = info.year.astype(int)
    info.affinity = info.affinity.astype(float)
    return info

df_pdb_id = pd.read_csv(
    '../index/INDEX_general_PL_name.2020', sep="  ", comment='#', header=None, engine='python',
    names=['pdb', 'year', 'uid', 'd', 'e','f','g','h','i','j','k','l','m','n','o']
)
df_pdb_id = df_pdb_id[['pdb','uid']]
data = read_pdbbind_data('../index/INDEX_general_PL_data.2020')
data = data.merge(df_pdb_id, on=['pdb'])

binding_affinity_dict =  dict(zip(data.pdb, data.affinity))

with open('../data/binding_affinity_dict.pkl', 'wb') as f:
    pickle.dump(binding_affinity_dict, f)
