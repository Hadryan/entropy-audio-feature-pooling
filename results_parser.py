import json
import os
import pandas as pd
from pathlib import Path

rootdir = "logs/speaker_noise_uniform_scale_0.25_train_F_val_F_test_T/checkpoints"

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



name	FFT x2 cros entr	FFT x2 acc	FFT x1 cros entr	FFT x1 acc	FFT x0.5 cros entr	FFT x0.5 acc	FFT x0.25 cros entr	FFT x0.25 acc
MMMM	&1.45	&0.62	&1.1	&0.71	&0.8	&0.78	&0.55	&0.86
MMEM	&1.38	&0.64	&1.09	&0.72	&0.79	&0.81	&0.59	&0.88
MMME	&1.51	&0.61	&1.15	&0.71	&0.8	&0.81	&0.58	&0.86
EMMM	&1.69	&0.51	&1.26	&0.68	&0.88	&0.81	&0.61	&0.88
MEMM	&1.44	&0.61	&1.1	&0.71	&0.79	&0.81	&0.58	&0.87
EMEM	&1.6	&0.56	&1.22	&0.69	&0.92	&0.78	&0.73	&0.84
MEEM	&1.34	&0.64	&1.04	&0.72	&0.76	&0.8	&0.56	&0.86
EMME	&1.41	&0.64	&1.06	&0.71	&0.75	&0.79	&0.54	&0.85
EEMM	&1.61	&0.54	&1.3	&0.65	&0.97	&0.75	&0.73	&0.83
EEME	&1.83	&0.58	&1.65	&0.62	&0.86	&0.76	&0.63	&0.83
EEEM	&1.41	&0.62	&1.09	&0.7	&0.8	&0.79	&0.62	&0.85
MEME	&1.8	&0.44	&1.37	&0.6	&0.97	&0.75	&0.71	&0.83
MMEE	&1.41	&0.63	&1.11	&0.7	&0.83	&0.77	&0.62	&0.84
EEEE	&1.43	&0.64	&1.22	&0.67	&0.97	&0.72	&0.76	&0.79
EMEE	&1.37	&0.63	&1.07	&0.71	&0.81	&0.78	&0.62	&0.84
MEEE	&1.58	&0.62	&1.26	&0.67	&0.92	&0.73	&0.66	&0.82


name & FFF  & FFT x0.25 &FFT x0.5 & FFT x1 & FFT x2 \\
MMMM & 0.932 & 0.856 & 0.782 & 0.706 & 0.620  \\
MMEM & 0.931 & 0.875 & 0.811 & 0.720 & 0.637  \\
MMME & 0.926 & 0.861 & 0.805 & 0.707 & 0.611  \\
EMMM & 0.923 & 0.876 & 0.805 & 0.677 & 0.512  \\
MEMM & 0.923 & 0.870 & 0.810 & 0.713 & 0.605  \\
EMEM & 0.923 & 0.836 & 0.775 & 0.686 & 0.556  \\
MEEM & 0.917 & 0.860 & 0.795 & 0.717 & 0.641  \\
EMME & 0.916 & 0.854 & 0.787 & 0.712 & 0.644  \\
EEMM & 0.916 & 0.826 & 0.748 & 0.645 & 0.540  \\
EEME & 0.915 & 0.830 & 0.755 & 0.624 & 0.581  \\
EEEM & 0.914 & 0.846 & 0.785 & 0.702 & 0.619  \\
MEME & 0.909 & 0.833 & 0.746 & 0.602 & 0.437  \\
MMEE & 0.907 & 0.843 & 0.773 & 0.696 & 0.634  \\
EEEE & 0.901 & 0.792 & 0.724 & 0.668 & 0.638  \\
EMEE & 0.898 & 0.841 & 0.776 & 0.708 & 0.628  \\
MEEE & 0.888 & 0.816 & 0.734 & 0.668 & 0.616  \\

name & FFF  & FFT x0.25 &FFT x0.5 & FFT x1 & FFT x2 \\
MMMM & 0.289 & 0.553 & 0.797 & 1.102 & 1.448  \\
MMEM & 0.321 & 0.587 & 0.791 & 1.088 & 1.380  \\
MMME & 0.328 & 0.583 & 0.797 & 1.151 & 1.513  \\
EMMM & 0.325 & 0.610 & 0.882 & 1.260 & 1.689  \\
MEMM & 0.341 & 0.580 & 0.786 & 1.097 & 1.441  \\
EMEM & 0.325 & 0.733 & 0.924 & 1.220 & 1.595  \\
MEEM & 0.362 & 0.563 & 0.764 & 1.040 & 1.344  \\
EMME & 0.346 & 0.539 & 0.746 & 1.058 & 1.406  \\
EEMM & 0.362 & 0.728 & 0.967 & 1.295 & 1.605  \\
EEME & 0.374 & 0.625 & 0.858 & 1.646 & 1.831  \\
EEEM & 0.365 & 0.617 & 0.804 & 1.089 & 1.408  \\
MEME & 0.411 & 0.708 & 0.967 & 1.367 & 1.800  \\
MMEE & 0.408 & 0.620 & 0.829 & 1.110 & 1.405  \\
EEEE & 0.405 & 0.757 & 0.973 & 1.224 & 1.434  \\
EMEE & 0.419 & 0.620 & 0.808 & 1.069 & 1.371  \\
MEEE & 0.456 & 0.663 & 0.924 & 1.258 & 1.582  \\
