"""
Script for generating plots.
"""
import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import argparse

from stepback.record import Record
from stepback.utils import get_output_filenames
from stepback.plotting import plot_stability, plot_step_sizes

################# Main setup ###############################
parser = argparse.ArgumentParser(description='Generate step-back plots.')
parser.add_argument('-i', '--id', nargs='?', type=str, default='test', help="The id of the config (its file name).")
args = parser.parse_args()


try:
    exp_id = args.id
    save = True
except:
    exp_id = 'cifar100_resnet110'
    save = False
outdir = 'output/plots/' + exp_id
if not os.path.isdir(outdir):
    os.makedirs(outdir)

output_names = get_output_filenames(exp_id)
############################################################

#%%
#%matplotlib qt5

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 13
plt.rcParams['axes.linewidth'] = 1
plt.rc('text', usetex=True)

#%%

R = Record(output_names)

# Rename method names everywhere (configs, id_df, base_df) before any plotting.
rename_map = {
    'nesgd-adam_infty-lmo-you': 'muon-You',
    'nesgd-adam_infty-lmo-polarexpress': 'muon-PolarExp',
    'nesgd-adam_infty-lmo-jordan': 'muon-Jordan',
    'nesgd-adam_infty-lmo-newton': 'muon-Newton',
}
for old, new in rename_map.items():
    if 'name' in R.base_df.columns:
        R.base_df.loc[R.base_df['name'] == old, 'name'] = new
    if 'name' in R.id_df.columns:
        R.id_df.loc[R.id_df['name'] == old, 'name'] = new
    if hasattr(R, 'configs'):
        for c in R.configs:
            if c.get('name') == old:
                c['name'] = new

# Apply filters after renaming
R.filter(drop={'name': ['momo-adam-star', 'momo-star']})
R.filter(drop={'name': ['adabelief', 'adabound', 'lion', 'prox-sps']})
R.filter(keep={'lr_schedule': 'constant'})

base_df = R.base_df
id_df   = R.id_df
# Ensure grouping for best runs uses renamed names
base_df['name'] = base_df['name'].replace(rename_map)
id_df['name'] = id_df['name'].replace(rename_map)

#%% plot training curves for a subset of runs:

# Re-apply rename to underlying Record dataframes in case filters reintroduced originals
R.base_df['name'] = R.base_df['name'].replace(rename_map)
R.id_df['name'] = R.id_df['name'].replace(rename_map)

# takes 3 best runs per methods (after renaming)
best = base_df[base_df.epoch == base_df.epoch.max()].groupby('name')['val_score'].nlargest(3)
ixx = base_df.id[best.index.levels[1]]
df1 = base_df.loc[base_df.id.isin(ixx), :].copy()

# Final safeguard rename for plotting (legend labels)
df1['name'] = df1['name'].replace(rename_map)

y0 = 0.3 if exp_id=='cifar100_resnet110' else 0.4 if exp_id=='cifar10_vit' else 0.6
fig, ax = R.plot_metric(df=df1, s='val_score', ylim=(y0, 1.05*df1.val_score.max()), log_scale=False, figsize=(4,3.5), legend=False)
fig.subplots_adjust(top=0.975,bottom=0.16,left=0.16,right=0.975)
if save:
    fig.savefig('output/plots/' + exp_id + f'/all_val_score.pdf')

fig, ax = R.plot_metric(df=df1, s='train_loss', log_scale=True, figsize=(4,3.5), legend=False)
fig.subplots_adjust(top=0.975,bottom=0.16,left=0.17,right=0.975)
if save:
    fig.savefig('output/plots/' + exp_id + f'/all_train_loss.pdf')


#%% stability plots
FIGSIZE = (5.2,3.4)
## Cifar100
fig, axs = plot_stability(R, score='val_score', xaxis='lr', sigma=1, legend=None, ylim= (0.2, 0.7), cutoff=None, figsize=FIGSIZE, save=save)
fig, axs = plot_stability(R, score='train_loss', xaxis='lr', sigma=1, legend=None,  cutoff=None, figsize=FIGSIZE, save=save)
## Cifar10
# fig, axs = plot_stability(R, score='val_score', xaxis='lr', sigma=1, legend=None,  cutoff=None, figsize=FIGSIZE, save=save)
# fig, axs = plot_stability(R, score='train_loss', xaxis='lr', sigma=1, legend=None,   cutoff=None, figsize=FIGSIZE, save=save)
# fig, axs = plot_stability(R, score='val_score', xaxis='lr', sigma=1, legend=None,  cutoff=None, figsize=FIGSIZE, save=save)
# fig, axs = plot_stability(R, score='train_loss', xaxis='lr', sigma=1, legend=None,  cutoff=None, figsize=FIGSIZE, save=save)
fig, axs = plot_stability(R, score=['train_loss', 'val_score'], xaxis='lr', sigma=1, legend=None, cutoff=None, figsize=(4.8,6.4), save=save)

# Early-epoch (1%) stability plots
early_pct = 0.025
if 'cifar100' in exp_id and early_pct == 0.025:
    fig, _ = plot_stability(R, score='val_score', xaxis='lr', sigma=1, legend=None, ylim=(0.0,0.5), at_pct=early_pct, figsize=FIGSIZE, save=save)
    fig, _ = plot_stability(R, score='train_loss', xaxis='lr', sigma=1, legend=None,ylim=(0.0,6), at_pct=early_pct, figsize=FIGSIZE, save=save)
else:
    fig, _ = plot_stability(R, score='val_score', xaxis='lr', sigma=1, legend=None, at_pct=early_pct, figsize=FIGSIZE, save=save)
    fig, _ = plot_stability(R, score='train_loss', xaxis='lr', sigma=1, legend=None, at_pct=early_pct, figsize=FIGSIZE, save=save)


# %%
