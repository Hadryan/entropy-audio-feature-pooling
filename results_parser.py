import json
import os
import pandas as pd
from pathlib import Path

rootdir = "logs/commands_noise_train_F_val_F_test_F/checkpoints"

p = Path(rootdir)
df_summary = None
cols_time = ["epoch",
             "global_step"]

cols = [
    "val_loss",
    "val_acc",
    "train_loss",
    "train_acc",
    "test_loss",
    "test_acc"
]

cols_sd = [
    "val_loss_sd",
    "val_acc_sd",
    "train_loss_sd",
    "train_acc_sd",
    "test_loss_sd",
    "test_acc_sd"
]

all_cols = cols_time + cols

def metadata2df(df_all):
    with sub.joinpath('metadata.json').open() as f:
        json_object = json.load(f)
        df = pd.DataFrame(json_object, index=[0])
        if df_all is not None:
            df_all = df_all.append(df, ignore_index=True)
        else:
            df_all = df
    return df_all


for child in p.iterdir():
    # print(child)
    df_all = None
    if child.is_dir():
        for sub in child.iterdir():
            # print(sub)
            df_all = metadata2df(df_all)

        df_all[all_cols] = df_all[all_cols].astype(float)
        df_mean = df_all[all_cols].mean()
        df_sd = df_all[cols].std()
        # df_mean['model_save_path'] = "mean"
        # df_sd['model_save_path'] = "sd"
        df_mean['name'] = child.name
        # df_sd['name'] = child.name
        df_mean = pd.DataFrame(df_mean).transpose()
        df_sd = pd.DataFrame(df_sd).transpose()
        df_sd.columns = cols_sd

        # df_all = df_all.append(df_mean, ignore_index=True)
        # df_all = df_all.append(df_sd, ignore_index=True)
        if df_summary is not None:
            # df_summary = df_summary.append(df_mean, ignore_index=True)
            # df_summary = df_summary.append(df_sd, ignore_index=True)
            concatenated = pd.concat([df_mean, df_sd.reindex(df_mean.index)], axis=1)
            df_summary = df_summary.append(concatenated, ignore_index=True)
        else:
            # df_summary = df_mean
            # df_summary = df_summary.append(df_sd, ignore_index=True)
            concatenated = pd.concat([df_mean, df_sd.reindex(df_mean.index)], axis=1)
            df_summary = concatenated

        # df_all.to_csv(child.joinpath('/stats.csv'))

print(df_summary.head())
df_summary.to_csv(p.joinpath('summary.csv'))

# json_object["epoch"]
# json_object["global_step"]
# json_object["val_loss"]
# json_object["val_acc"]
# json_object["train_loss"]
# json_object["train_acc"]
# json_object["test_loss"]
# json_object["test_acc"]
