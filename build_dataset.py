from tqdm.auto import tqdm
from rdkit import DataStructs
from utils import data_utils
from utils.data_utils import MolNetData
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdRascalMCES
from collections import Counter
import numpy as np
from multiprocessing import Pool, cpu_count
from func_timeout import func_timeout, FunctionTimedOut

import os
import sys
#PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.getcwd()
ACCFG_DIR = os.path.join(PROJECT_DIR, 'AccFG_private')
sys.path.append(ACCFG_DIR)
from accfg import (AccFG, draw_mol_with_fgs, molimg, set_atom_idx,
                   img_grid, compare_mols, draw_compare_mols,
                   draw_RascalMCES, print_fg_tree)
afg_lite = AccFG(print_load_info=True, lite=True)

from utils.fg_utils import *
import deepchem as dc
import argparse


def get_dataset(name):
    # Quantum Mechanics
    if name == 'qm7':
        tasks, datasets, transformers = dc.molnet.load_qm7(featurizer='GraphConv',splitter=None, transformers=['balancing'])
    elif name == 'qm8':
        tasks, datasets, transformers = dc.molnet.load_qm8(featurizer='GraphConv',splitter=None, transformers=['balancing'])
    elif name == 'qm9':
        tasks, datasets, transformers = dc.molnet.load_qm9(featurizer='GraphConv',splitter=None, transformers=['balancing'])
    # Physical Chemistry (Regression)
    elif name == 'esol':
        tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='GraphConv',splitter=None, transformers=['balancing'])
    elif name == 'lipo':
        tasks, datasets, transformers = dc.molnet.load_lipo(featurizer='GraphConv',splitter=None, transformers=['balancing'])
    elif name == 'freesolv':
        tasks, datasets, transformers = dc.molnet.load_freesolv(featurizer='GraphConv',splitter=None, transformers=['balancing'])
    # Biophysics
    elif name == 'hiv':
        tasks, datasets, transformers = dc.molnet.load_hiv(featurizer='GraphConv',splitter=None, transformers=['balancing'])
    elif name == 'bace':
        tasks, datasets, transformers = dc.molnet.load_bace_classification(featurizer='GraphConv',splitter=None, transformers=['balancing'])
    elif name =='muv': #17 tasks
        tasks, datasets, transformers = dc.molnet.load_muv(featurizer='GraphConv',splitter=None, transformers=['balancing'])
    # Physiology
    elif name == 'bbbp': #1 task
        tasks, datasets, transformers = dc.molnet.load_bbbp(featurizer='GraphConv',splitter=None, transformers=['balancing'])
    elif name == 'tox21': #12 tasks
        tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='GraphConv',splitter=None, transformers=['balancing'])
    #elif name == 'toxcast': #600 tasks
    #    tasks, datasets, transformers = dc.molnet.load_toxcast(featurizer='GraphConv',splitter=None)
    elif name == 'sider': #27 tasks
        tasks, datasets, transformers = dc.molnet.load_sider(featurizer='GraphConv',splitter=None, transformers=['balancing'])    
    elif name == 'clintox': #2 tasks
        tasks, datasets, transformers = dc.molnet.load_clintox(featurizer='GraphConv',splitter=None, transformers=['balancing'])
    return datasets[0]


def canonicalize_smiles(smiles):
    try:
        if '.' in smiles: # skip multi-component SMILES
            return None
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol)
    except:
        return None

def build_smiles_property_df(dataset):
    smiles_property_df = pd.DataFrame(dataset.ids, columns=['smiles'])
    for i in range(dataset.y.shape[1]):
        smiles_property_df[i] = dataset.y[:, i]
    smiles_property_df['smiles'] = smiles_property_df['smiles'].progress_apply(canonicalize_smiles)
    smiles_property_df = smiles_property_df.dropna()
    smiles_property_df_dedupl = smiles_property_df.drop_duplicates(subset=['smiles'])
    print(f'removed {len(smiles_property_df) - len(smiles_property_df_dedupl)} duplicates, and get {len(smiles_property_df_dedupl)} unique smiles')
    return smiles_property_df_dedupl

def get_similarity_df(df):
    assert 'smiles' in df.columns
    fpgen = AllChem.GetMorganGenerator(radius=2,fpSize=512)
    smi_list = df['smiles'].tolist()
    mol_list = [Chem.MolFromSmiles(smi) for smi in smi_list]
    fp_list = [fpgen.GetFingerprint(mol) for mol in mol_list]  
    similarity_df = pd.DataFrame(index=smi_list, columns=smi_list)
    for i in tqdm(range(len(smi_list)-1)):
        target_smi = smi_list[i]
        s = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[i+1:])
        similarity_df.loc[target_smi, smi_list[i+1:]] = s
    return similarity_df

def compare_mols_in_df(mol_1, mol_2, afg=afg_lite, similarityThreshold=0.2, canonical=False):
    try:
        fg_alkane_diff = compare_mols(mol_1, mol_2, afg, similarityThreshold, canonical)
        return fg_alkane_diff
    except:
        return None

def get_compare_df(df, threshold=0.7):
    similarity_df = get_similarity_df(df)
    condition = similarity_df > threshold
    row_indices, col_indices = np.where(condition)
    pairs = [(similarity_df.index[row], similarity_df.columns[col]) for row, col in zip(row_indices, col_indices)]

    compare_df = pd.DataFrame(columns=['smiles_pair'])
    compare_df['smiles_pair'] = pairs
    compare_df['fg_alkane_diff'] = compare_df['smiles_pair'].progress_apply(lambda x: compare_mols_in_df(x[0], x[1], afg_lite))
    compare_df = compare_df.dropna(subset=['fg_alkane_diff'])
    compare_df = compare_df[compare_df['fg_alkane_diff'] != (([],[]),([],[]))]
    compare_df['target_smiles'] = compare_df['smiles_pair'].apply(lambda x: x[0])
    compare_df['ref_smiles'] = compare_df['smiles_pair'].apply(lambda x: x[1])
    compare_df['target_diff'] = compare_df['fg_alkane_diff'].apply(lambda x: x[0])
    compare_df['ref_diff'] = compare_df['fg_alkane_diff'].apply(lambda x: x[1])
    compare_df.drop(columns=['smiles_pair','fg_alkane_diff'], inplace=True)
    print(f'Found {len(compare_df)} pairs of molecules with similarity > f{threshold}')
    return compare_df

def generate_fingerprints(df):
    assert 'smiles' in df.columns
    fpgen = AllChem.GetMorganGenerator(radius=2, fpSize=512)
    smi_list = df['smiles'].tolist()
    mol_list = [Chem.MolFromSmiles(smi) for smi in smi_list]
    fp_list = [fpgen.GetFingerprint(mol) for mol in mol_list]
    return smi_list, fp_list

def get_similar_pairs(smiles_list, fp_list, threshold=0.7):
    n = len(smiles_list)
    for i in tqdm(range(n - 1)):
        sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[i + 1:])
        for j, sim in enumerate(sims):
            if sim > threshold:
                yield (smiles_list[i], smiles_list[i + 1 + j])

def compare_large_dataset(df, threshold=0.7, name = None):
    smi_list, fp_list = generate_fingerprints(df)
    print("Finding similar molecule pairs...")
    similar_pairs = list(get_similar_pairs(smi_list, fp_list, threshold))
    print(f'Found {len(similar_pairs)} pairs of molecules with similarity > {threshold}')
    valid_count = 0
    similar_pairs_df = pd.DataFrame(similar_pairs, columns=["target_smiles", "ref_smiles"])
    similar_pairs_df.to_csv(f"data/molnet/{name}_similar_pairs.csv", index=False)
    print("Saved similar pairs to similar_pairs.csv")
    
    output_csv_path = f"data/molnet/{name}_compare.csv"
    with open(output_csv_path, 'w') as f:
        f.write("target_smiles,ref_smiles,target_diff,ref_diff\n")
        pbar = tqdm(similar_pairs, desc="Comparing molecules")
        for smi1, smi2 in pbar:
            fg_alkane_diff = compare_mols_in_df(smi1, smi2)
            if fg_alkane_diff and fg_alkane_diff != (([], []), ([], [])):
                valid_count += 1
                target_diff, ref_diff = fg_alkane_diff
                f.write(f'"{smi1}","{smi2}","{target_diff}","{ref_diff}"\n')
            pbar.set_postfix(valid=valid_count)
    print(f"Similarity comparison complete. Results saved to: {output_csv_path}")
    return None
    
def run(dataset_name, threshold=0.7):
    dataset = get_dataset(dataset_name)
    print(f'Building smiles property dataframe for {dataset_name}...')
    smiles_property_df = build_smiles_property_df(dataset)
    smiles_property_df.to_csv(f'data/molnet/{dataset_name}.csv', index=False)
    
    if len(smiles_property_df) < 10000:
        print(f'Building comparison dataframe for {dataset_name}...')
        compare_df = get_compare_df(smiles_property_df, threshold)
        compare_df.to_csv(f'data/molnet/{dataset_name}_compare.csv', index=False)
    else:
        print(f'Building large comparison dataframe for {dataset_name}...')
        compare_large_dataset(smiles_property_df, threshold, name=dataset_name)
    
    return None

def arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', nargs='+', default=[
        'esol', 'lipo', 'freesolv', 'hiv', 'bace', 'muv', 
        'bbbp', 'tox21', 'sider', 'clintox'
    ], help='list of dataset names')
    parser.add_argument('--threshold',default=0.7, type=float, help='threshold')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parser()
    for dataset_name in args.dataset:
        print(f'Processing {dataset_name}...')
        run(dataset_name, args.threshold)