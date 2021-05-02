import json
import wandb
import pandas as pd

MAX_SAMPLES = 1e6
STREAM_COLUMN = "Data_Stream"
ENTITY = "aman"
PROJECT = "LL4LM"

def get_experiment_data(run_id, entity=ENTITY, project=PROJECT):
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    # TODO: use run.scan_history when documentation is available
    logs = run.history(samples=MAX_SAMPLES).set_index("_step")
    stream_fpath = logs[STREAM_COLUMN][0]["path"]
    stream_file = run.file(stream_fpath).download(replace=True)
    stream = [
        (dataset_name, dataset_examples) \
        for dataset_name, dataset_examples in json.load(stream_file)["data"]
    ]
    return logs, stream, run.name 

def get_test_accuracies(logs, stream, unitask=False):
    test_columns = [f"test/{dataset_name}/accuracy" for dataset_name, _ in stream]
    if unitask:
        return pd.Series({col.split("/")[1]: logs[col].dropna().squeeze() for col in test_columns})
    return logs[test_columns].dropna(axis=0).rename(lambda col: col.split("/")[1], axis="columns")

def get_train_accuracies(logs, rolling_window=20, dataset_name=None):
    column = f"train/{dataset_name}/accuracy" if dataset_name else "train/accuracy"
    accuracy = logs[column].dropna(axis=0).squeeze()
    return accuracy.rolling(rolling_window, min_periods=1).mean().dropna()

def get_gradient_measurements(logs, stream, measurement="interference", rolling_window=20):
    column = f"gradient/{measurement}"
    if column in logs:
        values = logs[column].dropna()
    else:
        boundary = 0
        values = []
        for dataset_name, dataset_examples in stream:
            idx, col = f"{dataset_name}_examples_seen", f"gradient/{dataset_name}/{measurement}"
            value = logs.set_index(idx)[col].dropna()
            value.index = value.index + boundary
            values.append(value)
            boundary += dataset_examples
        values = pd.concat(values)
    return values[values>0].rolling(rolling_window, min_periods=1).mean().dropna()
