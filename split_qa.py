import pandas as pd
from sklearn.model_selection import train_test_split
import glob

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
            ]
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

def split_qa_dataset(dataset_name):
    if dataset_name in regression_dataset_dict:
        task_list = regression_dataset_dict[dataset_name]
        task_type = 'regression'
    elif dataset_name in classification_dataset_dict:
        task_list = classification_dataset_dict[dataset_name]
        task_type = 'classification'
        # task_classes = ['comparison_bool', 'comparison_value', 'single_bool', 'single_value', 'interaction_bool', 'interaction_value']
    data_df = pd.read_json(f'./data/fgbench_qa/{dataset_name}.jsonl', lines=True)
    data_df['split'] = 'train'
    ### TODO: based on single and inter to get comp pairs!!!
    for dimension in ['single', 'interaction']:
        mol_df = data_df.loc[(data_df['type'] == f'{dimension}_bool_{task_type}') & (data_df['task_num'] == 0)]
        dataset_size = len(mol_df)
        if dataset_size > 50:
            test_size = 25
        else:
            test_size = int(dataset_size / 2)
        _, test_mol_df = train_test_split(mol_df, test_size=test_size, random_state=42)
        # Create a set of (target_smiles, ref_smiles) pairs from test_mol_df
        test_pairs = set(zip(test_mol_df['target_smiles'], test_mol_df['ref_smiles']))
        
        # Loop through data_df and mark rows as test only if they contain the same molecule pair
        for idx, row in data_df.iterrows():
            if (row['target_smiles'], row['ref_smiles']) in test_pairs:
                data_df.at[idx, 'split'] = 'test'
    
    # Save the split data
    test_df = data_df[data_df['split'] == 'test']
    train_df = data_df[data_df['split'] == 'train']
    # Save to files
    test_df.to_json(f'./data/fgbench_qa/{dataset_name}_test.jsonl', orient='records', lines=True)
    train_df.to_json(f'./data/fgbench_qa/{dataset_name}_train.jsonl', orient='records', lines=True)
    print(dataset_name, len(test_df), len(train_df))
    return test_df, train_df
        
def run():
    all_test_df = pd.DataFrame()
    all_train_df = pd.DataFrame()
    all_dataset_names = list(regression_dataset_dict.keys()) + list(classification_dataset_dict.keys())
    for dataset_name in all_dataset_names:
        test_df, train_df = split_qa_dataset(dataset_name)
        all_test_df = pd.concat([all_test_df, test_df], ignore_index=True)
        all_train_df = pd.concat([all_train_df, train_df], ignore_index=True)
    # Convert all columns to string to ensure compatibility
    for col in all_test_df.columns:
        all_test_df[col] = all_test_df[col].astype(str)
    for col in all_train_df.columns:
        all_train_df[col] = all_train_df[col].astype(str)
    # Save the combined dataframes to JSONL files
    all_test_df.to_json(f'./data/fgbench/test.jsonl', orient='records', lines=True)
    all_train_df.to_json(f'./data/fgbench/train.jsonl', orient='records', lines=True)

if __name__ == '__main__':
    #load_all_qa_dataset()
    run()
    # split_qa_dataset('esol')
    '''
    split_qa_dataset('lipo')
    split_qa_dataset('freesolv')
    split_qa_dataset('bace')
    split_qa_dataset('bbbp')
    split_qa_dataset('tox21')
    split_qa_dataset('sider')
    split_qa_dataset('clintox')
    '''