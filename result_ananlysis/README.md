# Analyze FGBench results

Use Python 3.10+ and install the dependency, then run from the repository root:

```bash
python result_ananlysis/analysis.py
```

Provide the seven open-source prediction files in `data/results/`, the closed-source
files `benchmark_result_4o.jsonl` and `benchmark_result_o3_mini.jsonl` in
`data/gpt_request/`. Use `--results-dir` or `--gpt-results-dir` to specify
alternative prediction directories.

Load ground truth from the `test` split of
[xuan-liu/FGBench](https://huggingface.co/datasets/xuan-liu/FGBench) using
`datasets.load_dataset`. Preserve the returned row order for batch
`custom_id` alignment.

Read `result_ananlysis/output/result.csv` for ACC, RMSE, and validity for each model and task, rounded to three decimals. Use `--output-dir PATH` to choose another destination.

To evaluate selected models, run:

```bash
python result_ananlysis/analysis.py --models llama-3.1-8b qwen nach0
```
