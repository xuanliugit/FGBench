### single questions
single_bool_classification_question = '''
For a molecule whose SMILES with atom number is {target_mapped_smiles}, its label for the property of {property_name} is {target_label}.

After modifying the molecule {edit_text}

Does the property of the modified molecule change? Your final answer should be 'True' or 'False'.
'''

single_bool_regression_question = '''
For a molecule whose SMILES with atom number is {target_mapped_smiles}, its label for the property of {property_name} is {target_label}.

After modifying the molecule {edit_text}

Does the property of the modified molecule increase? Your final answer should be 'True' or 'False'.
'''

single_value_regression_question = '''
For a molecule whose SMILES with atom number is {target_mapped_smiles}, its label for the property of {property_name} is {target_label}.

After modifying the molecule {edit_text}

What is the value change of the property for the modified molecule? Your final answer should be "[value]" for increase or "-[value]" for decrease.
'''

### interaction questions
interaction_bool_classification_question ='''
For a molecule whose SMILES with atom number is {target_mapped_smiles}, its label for the property of {property_name} is {target_label}.

After modifying the molecule {edit_text}

Does the property of the modified molecule change? Your final answer should be 'True' or 'False'.
'''

interaction_bool_regression_question = '''
For a molecule whose SMILES with atom number is {target_mapped_smiles}, its label for the property of {property_name} is {target_label}. 

After modifying the molecule {edit_text}

Does the property of the modified molecule increase? Your final answer should be 'True' or 'False'.
'''

interaction_value_regression_question = '''
For a molecule whose SMILES with atom number is {target_mapped_smiles}, its label for the property of {property_name} is {target_label}. 

After modifying the molecule {edit_text}

What is the value change of {property_name} for the modified molecule? Your final answer should be "[value]" for increase or "-[value]" for decrease.
'''

# Comparison
comparison_bool_classification_question = '''
For a target molecule whose SMILES is {target_smiles} and a reference molecule whose SMILES is {ref_smiles}, the reference molecule has the label for the property of {property_name} as {ref_label}. Does the target molecule have a different property label compared to the reference molecule? Your final answer should be 'True' or 'False'.
'''

comparison_bool_regression_question = '''
For a target molecule whose SMILES is {target_smiles} and a reference molecule whose SMILES is {ref_smiles}, the reference molecule has the label for the property of {property_name} as {ref_label}. Does the target molecule have a higher value of the property compared to the reference molecule? Your final answer should be 'True' or 'False'.
'''

comparison_value_regression_question = '''
For a target molecule whose SMILES is {target_smiles} and a reference molecule whose SMILES is {ref_smiles}, the reference molecule has the value for the property of {property_name} as {ref_label}. What is the value change of {property_name} for the target molecule compared to the reference molecule? Your final answer should be "[value]" for increase or "-[value]" for decrease.
'''

