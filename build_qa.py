from tqdm.auto import tqdm
from rdkit import DataStructs
from utils import data_utils
from utils.data_utils import MolNetData, get_iupac_name
import pandas as pd
from rdkit import Chem
from collections import Counter, defaultdict
import numpy as np
import argparse

import os
import sys
#PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.getcwd()
ACCFG_DIR = os.path.join(PROJECT_DIR, 'AccFG_private')
sys.path.append(ACCFG_DIR)

from accfg import (AccFG, draw_mol_with_fgs, molimg, set_atom_idx,
                   img_grid, compare_mols, draw_compare_mols,
                   draw_RascalMCES, print_fg_tree, remove_atoms_from_mol,
                   SmilesMCStoGridImage, remove_fg_list_from_mol, get_RascalMCES,
                   remove_atoms_add_hs, get_outer_bond_from_fg_list)
afg_lite = AccFG(print_load_info=True, lite=True)

from utils.fg_utils import *
from prompts.question import (single_bool_classification_question, single_bool_regression_question, single_value_regression_question, 
                               interaction_bool_classification_question, interaction_bool_regression_question, interaction_value_regression_question, 
                               comparison_bool_classification_question, comparison_bool_regression_question, comparison_value_regression_question)

def merge_diff_tuple(diff_tuple_list):
    merged_diff = []
    for diff_tuple in diff_tuple_list:
        merged_diff += diff_tuple
    return merged_diff
def exam_comparison(target_smiles, ref_smiles, target_diff, ref_diff):
    target_mol = Chem.MolFromSmiles(target_smiles)
    ref_mol = Chem.MolFromSmiles(ref_smiles)

    target_mol = set_atom_idx(target_mol,'atomNote')
    ref_mol = set_atom_idx(ref_mol, 'atomNote')
    target_fg_diff = merge_diff_tuple(target_diff)
    ref_fg_diff = merge_diff_tuple(ref_diff)
    target_remain_mol = remove_fg_list_from_mol(target_mol, target_fg_diff)
    ref_remain_mol = remove_fg_list_from_mol(ref_mol, ref_fg_diff)
    Chem.SanitizeMol(target_remain_mol)
    Chem.SanitizeMol(ref_remain_mol)
    print(Chem.MolToSmiles(target_remain_mol, isomericSmiles=False))
    print(Chem.MolToSmiles(ref_remain_mol, isomericSmiles=False))
    return  Chem.MolToSmiles(target_remain_mol, isomericSmiles=False) == Chem.MolToSmiles(ref_remain_mol, isomericSmiles=False)

def sort_bond_tuple(bond_tuple):
    return tuple(sorted(bond_tuple))

def get_frag_name_smi_from_atom(fg_tuple_list, atom_idx):
    for fg_name, fg_smiles, fg_atoms, outer_bonds, root_bond in fg_tuple_list:
        if atom_idx in fg_atoms:
            return f'{fg_name} ({fg_smiles})'
    return None

def rebuild_from_comparison(target_smiles, ref_smiles, target_diff, ref_diff):
    target_mol = Chem.MolFromSmiles(target_smiles)
    ref_mol = Chem.MolFromSmiles(ref_smiles)

    target_mol = set_atom_idx(target_mol,'atomNote')
    ref_mol = set_atom_idx(ref_mol, 'atomNote')
    
    target_mol_with_mapped_atom = Chem.MolFromSmiles(target_smiles)
    target_mol_with_mapped_atom = set_atom_idx(target_mol_with_mapped_atom, 'molAtomMapNumber')
    target_mapped_smiles = Chem.MolToSmiles(target_mol_with_mapped_atom)
    
    ref_mol_with_mapped_atom = Chem.MolFromSmiles(ref_smiles)
    ref_mol_with_mapped_atom = set_atom_idx(ref_mol_with_mapped_atom, 'molAtomMapNumber')
    ref_mapped_smiles = Chem.MolToSmiles(ref_mol_with_mapped_atom)
    
    target_fg_diff = merge_diff_tuple(target_diff)
    disconnect_list = []
    for fg_name, _, fg_atoms in target_fg_diff:
        disconnect_list.append((fg_name, fg_atoms))
    
    ref_fg_diff = merge_diff_tuple(ref_diff)
    target_remain_mol = remove_fg_list_from_mol(target_mol, target_fg_diff)
    ref_remain_mol = remove_fg_list_from_mol(ref_mol, ref_fg_diff)
    Chem.SanitizeMol(target_remain_mol)
    Chem.SanitizeMol(ref_remain_mol)
    target_smi = Chem.MolToSmiles(target_remain_mol)
    ref_smi = Chem.MolToSmiles(ref_remain_mol)
    if Chem.MolToSmiles(target_remain_mol, isomericSmiles=False) != Chem.MolToSmiles(ref_remain_mol, isomericSmiles=False):
        return None
    res_on_common_structure = get_RascalMCES(target_remain_mol, ref_remain_mol)
    atom_matches = res_on_common_structure[0].atomMatches()
    atom_matches_on_note = []
    for taget_idx, ref_idx in atom_matches:
        target_note = int(target_remain_mol.GetAtomWithIdx(taget_idx).GetProp('atomNote'))
        ref_note = int(ref_remain_mol.GetAtomWithIdx(ref_idx).GetProp('atomNote'))
        atom_matches_on_note.append((target_note, ref_note))
    target_to_ref_note_map = dict(atom_matches_on_note)
    ref_to_target_note_map = {ref_note: target_note for target_note, ref_note in atom_matches_on_note}
    ref_note_mapped_list = list(ref_to_target_note_map.keys())
    
    ref_fg_list_outer_bonds = get_outer_bond_from_fg_list(ref_mol, ref_fg_diff)


    fg_tuple_list = []
    unique_bonds = []
    unique_fg_smi = []
    for fg_name, n, fg_list, outer_bonds_list in ref_fg_list_outer_bonds:
        for fg_atoms, outer_bonds in zip(fg_list, outer_bonds_list):
            fg_smiles = Chem.MolFragmentToSmiles(ref_mol_with_mapped_atom,fg_atoms)
            unique_fg_smi.append(fg_smiles)
            root_bond = []
            for outer_bond in outer_bonds:
                outer_atom = outer_bond[1]
                if outer_atom in ref_note_mapped_list:
                    root_bond.append(outer_bond)
                unique_bonds.append(sort_bond_tuple(outer_bond))
            fg_tuple_list.append((fg_name, fg_smiles, fg_atoms, outer_bonds, root_bond))
    unique_bonds = set(unique_bonds)
    unique_fg_smi = set(unique_fg_smi)
    fg_tuple_list = sorted(fg_tuple_list, key=lambda x: x[-1], reverse=True)
    
    connect_list = []
    connect_dict = defaultdict(list)
    for fg_name, fg_smiles, fg_atoms, outer_bonds, root_bond in fg_tuple_list:
        if fg_smiles in unique_fg_smi:
            unique_fg_smi.discard(fg_smiles)
            for outer_bond in outer_bonds:
                if sort_bond_tuple(outer_bond) in unique_bonds:
                    unique_bonds.discard(sort_bond_tuple(outer_bond))
                    in_atom = outer_bond[0]
                    out_atom = outer_bond[1]
                    if out_atom in ref_note_mapped_list:
                        out_atom = ref_to_target_note_map[out_atom]
                        out_frag = 'target molecule'
                    else:
                        out_frag = get_frag_name_smi_from_atom(fg_tuple_list, out_atom)
                    connect_list.append((fg_name, fg_smiles, in_atom, out_atom, out_frag))
                    connect_dict[f'{fg_name} ({fg_smiles})'].append((in_atom, out_atom, out_frag))
    result = {'target_smiles': target_smiles,
              'target_mapped_smiles': target_mapped_smiles,
              'ref_smiles': ref_smiles,
              'ref_mapped_smiles': ref_mapped_smiles,
              'target_diff': target_diff,
              'ref_diff': ref_diff,
              'disconnect_list': disconnect_list,
              'connect_dict': connect_dict}
    return result

def load_dataset(dataset_name):
    df = pd.read_csv(f'data/molnet/{dataset_name}.csv')
    compare_df = pd.read_csv(f'data/molnet/{dataset_name}_compare.csv')
    compare_df['target_diff'] = compare_df['target_diff'].apply(lambda x: eval(x))
    compare_df['ref_diff'] = compare_df['ref_diff'].apply(lambda x: eval(x))
    return df, compare_df

def build_edit_text(disconnect_list, connect_dict):
    if disconnect_list and connect_dict:
        removed_fgs = '\n'.join([f'* removing {fg_name} at position {fg_atoms}' for fg_name, fg_atoms in disconnect_list])
        added_fgs_list = []
        for fg_name,atom_change_list in connect_dict.items():
            added_text = f'* adding {fg_name} by ' + ', '.join([f'connecting its position {in_atom} to the position {out_atom} of {out_frag}' for in_atom, out_atom, out_frag in atom_change_list])
            added_fgs_list.append(added_text)
        added_fgs = '\n'.join(added_fgs_list)
        return f'by removing the following functional groups: \n{removed_fgs} \nand adding the following functional groups: \n{added_fgs}'
    
    if disconnect_list and not connect_dict:
        removed_fgs = '\n'.join([f'* removing {fg_name} at postion {fg_atoms}' for fg_name, fg_atoms in disconnect_list])
        return f'by removing the following functional groups: \n{removed_fgs}'
    
    if not disconnect_list and connect_dict:
        added_fgs_list = []
        for fg_name,atom_change_list in connect_dict.items():
            added_text = f'* adding {fg_name} by ' + ', '.join([f'connecting its position {in_atom} to the position {out_atom} of {out_frag}' for in_atom, out_atom, out_frag in atom_change_list])
            added_fgs_list.append(added_text)
        added_fgs = '\n'.join(added_fgs_list)
        return f'by adding the following functional groups: \n{added_fgs}'

def filter_one_fg(row):
    target_diff = row['target_diff']
    ref_diff = row['ref_diff']
    target_fg_diff = merge_diff_tuple(target_diff)
    ref_fg_diff = merge_diff_tuple(ref_diff)
    empty_target_diff = len(target_fg_diff) == 0
    empty_ref_diff = len(ref_fg_diff) == 0
    one_target_fg = len(target_fg_diff) == 1
    one_ref_fg = len(ref_fg_diff) == 1
    only_one_fg = (one_target_fg and empty_ref_diff) or (one_ref_fg and empty_target_diff)
    return only_one_fg


def run(dataset_name, task_list, tag):
    df, compare_df = load_dataset(dataset_name)
    compare_info_dict_list = []
    for i in tqdm(range(len(compare_df))):
        target_smiles = compare_df.loc[i, 'target_smiles']
        ref_smiles = compare_df.loc[i, 'ref_smiles']
        target_diff = compare_df.loc[i, 'target_diff']
        ref_diff = compare_df.loc[i, 'ref_diff']
        try:
            compare_info_dict = rebuild_from_comparison(target_smiles, ref_smiles, target_diff, ref_diff)
        except:
            compare_info_dict = None
        if compare_info_dict is None:
            continue
        for task_num, task in enumerate(task_list):
            new_compare_info_dict = compare_info_dict.copy()
            new_compare_info_dict['target_label'] = df[df['smiles'] == target_smiles][str(task_num)].values[0]
            new_compare_info_dict['ref_label'] = df[df['smiles'] == ref_smiles][str(task_num)].values[0]
            new_compare_info_dict['property_name'] = task
            
            compare_info_dict_list.append(new_compare_info_dict)
    compare_info_df = pd.DataFrame(compare_info_dict_list)
    compare_info_df.to_csv(f'data/molnet/{dataset_name}_compare_info.csv', index=False)
    # get task specific df and build Q&A
    
    qa_df = pd.DataFrame()
    
    for task_num, task in enumerate(task_list):
        task_df = compare_info_df.loc[compare_info_df['property_name'] == task]
        single_fg_df = task_df[task_df.apply(filter_one_fg, axis=1)]   
        interaction_fg_df = task_df[~task_df.apply(filter_one_fg, axis=1)]
        print(f'Processing {task}... len(task_df): {len(task_df)} len(single_fg_df): {len(single_fg_df)}, len(interaction_fg_df): {len(interaction_fg_df)}')
        if tag == 'classification':
            sbc_list = []
            ibc_list = []
            cbc_list = []
            for i in tqdm(range(len(single_fg_df))):
                row = single_fg_df.iloc[i]
                edit_text = build_edit_text(row['disconnect_list'], row['connect_dict'])
                sbc_question = single_bool_classification_question.format(target_mapped_smiles=row['target_mapped_smiles'],
                                                                    property_name=row['property_name'],
                                                                    target_label='True' if row['target_label']==1 else 'False',
                                                                    edit_text=edit_text)
                sbc_answer = 'True' if row['target_label'] != row['ref_label'] else 'False'
                sbc_list.append({'question': sbc_question, 'answer': sbc_answer} | row.to_dict()) # include Q&A and metadata
            
            for i in tqdm(range(len(interaction_fg_df))):
                row = interaction_fg_df.iloc[i]
                edit_text = build_edit_text(row['disconnect_list'], row['connect_dict'])
                ibc_question = interaction_bool_classification_question.format(target_mapped_smiles=row['target_mapped_smiles'],
                                                                    property_name=row['property_name'],
                                                                    target_label='True' if row['target_label']==1 else 'False',
                                                                    edit_text=edit_text)
                ibc_answer = 'True' if row['target_label'] != row['ref_label'] else 'False'
                ibc_list.append({'question': ibc_question, 'answer': ibc_answer} | row.to_dict())
                
            for i in tqdm(range(len(task_df))):
                row = task_df.iloc[i]
                edit_text = build_edit_text(row['disconnect_list'], row['connect_dict'])
                cbc_question = comparison_bool_classification_question.format(target_smiles=row['target_smiles'],
                                                                    ref_smiles=row['ref_smiles'],
                                                                    property_name=row['property_name'],
                                                                    ref_label='True' if row['ref_label']==1 else 'False')
                cbc_answer = 'True' if row['target_label'] != row['ref_label'] else 'False'
                cbc_list.append({'question': cbc_question, 'answer': cbc_answer} | row.to_dict())
            sbc_df = pd.DataFrame(sbc_list)
            ibc_df = pd.DataFrame(ibc_list)
            cbc_df = pd.DataFrame(cbc_list)
            # Concatenate all classification dataframes for this task
            subtask_qa_df = pd.concat([
                sbc_df.assign(type='single_bool_classification'),
                ibc_df.assign(type='interaction_bool_classification'),
                cbc_df.assign(type='comparison_bool_classification')
            ])

            # Add task information to the dataframe
            subtask_qa_df['dataset'] = dataset_name
            subtask_qa_df['task_num'] = task_num

            # Concatenate to the overall qa_df
            qa_df = pd.concat([qa_df, subtask_qa_df], ignore_index=True)
            
            sbc_df.to_csv(f'data/molnet_qa/{dataset_name}_{task_num}_single_bool_classification.csv', index=False)
            ibc_df.to_csv(f'data/molnet_qa/{dataset_name}_{task_num}_interaction_bool_classification.csv', index=False)
            cbc_df.to_csv(f'data/molnet_qa/{dataset_name}_{task_num}_comparison_bool_classification.csv', index=False)

        elif tag == 'regression':
            ibr_list = []
            ivr_list = []
            cbr_list = []
            cvr_list = []
            sbr_list = []
            svr_list = []
            for i in tqdm(range(len(single_fg_df))):
                row = single_fg_df.iloc[i]
                edit_text = build_edit_text(row['disconnect_list'], row['connect_dict'])
                sbr_question = single_bool_regression_question.format(target_mapped_smiles=row['target_mapped_smiles'],
                                                                    property_name=row['property_name'],
                                                                    target_label=round(row['target_label'],3),
                                                                    edit_text=edit_text)
                sbr_answer = 'True' if row['target_label'] < row['ref_label'] else 'False'
                sbr_list.append({'question': sbr_question, 'answer': sbr_answer} | row.to_dict())
                
                svr_question = single_value_regression_question.format(target_mapped_smiles=row['target_mapped_smiles'],
                                                                    property_name=row['property_name'],
                                                                    target_label=round(row['target_label'],3),
                                                                    edit_text=edit_text)
                svr_answer = row["ref_label"] - row["target_label"]
                svr_list.append({'question': svr_question, 'answer': svr_answer} | row.to_dict())
                
            for i in tqdm(range(len(interaction_fg_df))):
                row = interaction_fg_df.iloc[i]
                edit_text = build_edit_text(row['disconnect_list'], row['connect_dict'])
                
                ibr_question = interaction_bool_regression_question.format(target_mapped_smiles=row['target_mapped_smiles'],
                                                                    property_name=row['property_name'],
                                                                    target_label=round(row['target_label'],3),
                                                                    edit_text=edit_text)
                ibr_answer = 'True' if row['target_label'] < row['ref_label'] else 'False'
                ibr_list.append({'question': ibr_question, 'answer': ibr_answer} | row.to_dict())
                
                ivr_question = interaction_value_regression_question.format(target_mapped_smiles=row['target_mapped_smiles'],
                                                                    property_name=row['property_name'],
                                                                    target_label=round(row['target_label'],3),
                                                                    edit_text=edit_text)
                ivr_answer = row["ref_label"] - row["target_label"]
                ivr_list.append({'question': ivr_question, 'answer': ivr_answer} | row.to_dict())
                
            for i in tqdm(range(len(task_df))):
                row = task_df.iloc[i]
                edit_text = build_edit_text(row['disconnect_list'], row['connect_dict'])
                
                cbr_question = comparison_bool_regression_question.format(target_smiles=row['target_smiles'],
                                                                    ref_smiles=row['ref_smiles'],
                                                                    property_name=row['property_name'],
                                                                    ref_label=round(row['ref_label'],3))
                cbr_answer = 'True' if row['target_label'] > row['ref_label'] else 'False'
                cbr_list.append({'question': cbr_question, 'answer': cbr_answer} | row.to_dict())
                
                cvr_question = comparison_value_regression_question.format(target_smiles=row['target_smiles'],
                                                                    ref_smiles=row['ref_smiles'],
                                                                    property_name=row['property_name'],
                                                                    ref_label=round(row['ref_label'],3))
                cvr_answer = row["target_label"] - row["ref_label"]
                cvr_list.append({'question': cvr_question, 'answer': cvr_answer} | row.to_dict())

            ibr_df = pd.DataFrame(ibr_list)
            ivr_df = pd.DataFrame(ivr_list)
            cbr_df = pd.DataFrame(cbr_list)
            cvr_df = pd.DataFrame(cvr_list)
            sbr_df = pd.DataFrame(sbr_list)
            svr_df = pd.DataFrame(svr_list)
            # Concatenate all regression dataframes for this task
            subtask_qa_df = pd.concat([
                sbr_df.assign(type='single_bool_regression'),
                svr_df.assign(type='single_value_regression'),
                ibr_df.assign(type='interaction_bool_regression'),
                ivr_df.assign(type='interaction_value_regression'),
                cbr_df.assign(type='comparison_bool_regression'),
                cvr_df.assign(type='comparison_value_regression')
            ])

            # Add task information to the dataframe
            subtask_qa_df['dataset'] = dataset_name
            subtask_qa_df['task_num'] = task_num

            # Concatenate to the overall qa_df
            qa_df = pd.concat([qa_df, subtask_qa_df], ignore_index=True)
            
            ibr_df.to_csv(f'data/molnet_qa/{dataset_name}_{task_num}_interaction_bool_regression.csv', index=False)
            ivr_df.to_csv(f'data/molnet_qa/{dataset_name}_{task_num}_interaction_value_regression.csv', index=False)
            cbr_df.to_csv(f'data/molnet_qa/{dataset_name}_{task_num}_comparison_bool_regression.csv', index=False)
            cvr_df.to_csv(f'data/molnet_qa/{dataset_name}_{task_num}_comparison_value_regression.csv', index=False) 
            sbr_df.to_csv(f'data/molnet_qa/{dataset_name}_{task_num}_single_bool_regression.csv', index=False)
            svr_df.to_csv(f'data/molnet_qa/{dataset_name}_{task_num}_single_value_regression.csv', index=False)
    
    qa_df.to_json(f'data/fgbench_qa/{dataset_name}.jsonl', orient='records', lines=True)
    return None

regression_dataset_dict = {
    'esol':['log-scale water solubility in mols per litre'],
    'lipo':['octanol/water distribution coefficient (logD at pH 7.4)'],
    'freesolv':['hydration free energy in water'],
    'qm9':[
            'Dipole moment (unit: D)',
            'Isotropic polarizability (unit: Bohr^3)',
            'Highest occupied molecular orbital energy (unit: Hartree)',
            'Lowest unoccupied molecular orbital energy (unit: Hartree)',
            'Gap between HOMO and LUMO (unit: Hartree)',
            'Electronic spatial extent (unit: Bohr^2)',
            'Zero point vibrational energy (unit: Hartree)',
            'Heat capavity at 298.15K (unit: cal/(mol*K))',
            'Internal energy at 0K (unit: Hartree)',
            'Internal energy at 298.15K (unit: Hartree)',
            'Enthalpy at 298.15K (unit: Hartree)',
            'Free energy at 298.15K (unit: Hartree)'
            ] #12
}

classification_dataset_dict = {
    # Biophysics
    'hiv':['HIV inhibitory activity'], #1
    'bace': ['human β-secretase 1 (BACE-1) inhibitory activity'], #1
    # Physiology
    'bbbp': ['blood-brain barrier penetration'], #1
    'tox21': [
                "Androgen receptor pathway activation",
                "Androgen receptor ligand-binding domain activation",
                "Aryl hydrocarbon receptor activation",
                "Inhibition of aromatase enzyme",
                "Estrogen receptor pathway activation",
                "Estrogen receptor ligand-binding domain activation",
                "Activation of peroxisome proliferator-activated receptor gamma",
                "Activation of antioxidant response element signaling",
                "Activation of ATAD5-mediated DNA damage response",
                "Activation of heat shock factor response element signaling",
                "Disruption of mitochondrial membrane potential",
                "Activation of p53 tumor suppressor pathway"
            ], #12
    'sider': [
                "Cause liver and bile system disorders",
                "Cause metabolic and nutritional disorders",
                "Cause product-related issues",
                "Cause eye disorders",
                "Cause abnormal medical test results",
                "Cause muscle, bone, and connective tissue disorders",
                "Cause gastrointestinal disorders",
                "Cause adverse social circumstances",
                "Cause immune system disorders",
                "Cause reproductive system and breast disorders",
                "Cause tumors and abnormal growths (benign, malignant, or unspecified)",
                "Cause general disorders and administration site conditions",
                "Cause endocrine (hormonal) disorders",
                "Cause complications from surgical and medical procedures",
                "Cause vascular (blood vessel) disorders",
                "Cause blood and lymphatic system disorders",
                "Cause skin and subcutaneous tissue disorders",
                "Cause congenital, familial, and genetic disorders",
                "Cause infections and infestations",
                "Cause respiratory and chest disorders",
                "Cause psychiatric disorders",
                "Cause renal and urinary system disorders",
                "Cause complications during pregnancy, childbirth, or perinatal period",
                "Cause ear and balance disorders",
                "Cause cardiac disorders",
                "Cause nervous system disorders",
                "Cause injury, poisoning, and procedural complications"
            ], #27
    'clintox': ['drugs approved by the FDA and passed clinical trials'] # 1 task
    }

def arg_praser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', nargs='+', default=[
        'esol', 'lipo', 'freesolv', 'qm9', 'bace', 'hiv',
        'bbbp', 'tox21', 'sider', 'clintox'
    ], help='list of dataset names')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_praser()
    for dataset_name in args.dataset:
        print(f'Processing {dataset_name}...')
        if dataset_name in regression_dataset_dict:
            task_list = regression_dataset_dict[dataset_name]
            run(dataset_name, task_list, 'regression')
        elif dataset_name in classification_dataset_dict:
            task_list = classification_dataset_dict[dataset_name]
            run(dataset_name, task_list, 'classification')
        else:
            print(f'No task list for {dataset_name}')
            continue
        
