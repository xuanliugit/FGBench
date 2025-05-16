<div align="center">

# FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models


[![Website](https://img.shields.io/badge/Website-%F0%9F%8C%8D-blue?style=for-the-badge&logoWidth=40)](https://github.com/xuanliugit/FGBench)
[![Dataset](https://img.shields.io/badge/Dataset-%F0%9F%92%BE-green?style=for-the-badge&logoWidth=40)](https://huggingface.co/datasets/xuan-liu/FGBench/)


</div>

## Quick start
### Usage 

```
from datasets import load_dataset
dataset = load_dataset("xuan-liu/FGBench") # Loading all 

dataset_test = load_dataset("xuan-liu/FGBench", split = "test") # Benchmark dataset

dataset_train = load_dataset("xuan-liu/FGBench", split = "train")
```

This dataset is constructed with functional group information based on MoleculeNet dataset. The datasets and tasks used in FGBench are listed below.

```
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
```