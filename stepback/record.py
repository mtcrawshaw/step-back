from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
import numpy as np
import copy
import itertools
import os
from typing import Union, Optional, Tuple
import warnings
from pandas.api.types import is_numeric_dtype

from .log import Container
from .defaults import DEFAULTS


SCORE_NAMES = {'train_loss': 'Training loss', 
               'val_loss': 'Validation loss', 
               'train_score': 'Training score', 
               'val_score': 'Validation score',
               'model_norm': r'$\|x^k\|$',
               'grad_norm': r'$\|g_k\|$',
               'fstar': r'$f_*^k$'
}

AES = { 'sgd':              {'color': "#f68427", 'markevery': 15, 'zorder': 7},
        'sgd-m':            {'color': '#de9151', 'markevery': 8, 'zorder': 8},
        'adam':             {'color': '#f34213', 'markevery': 10, 'zorder': 9}, 
        'adamw':            {'color': '#f34213', 'markevery': 10, 'zorder': 9},
        'momo':             {'color': '#023047', 'markevery': 5, 'zorder': 11},
        'momo-adam':        {'color': '#3F88C5', 'markevery': 6, 'zorder': 10},
        'momo-star':        {'color': '#87b37a', 'markevery': 3, 'zorder': 13},
        'momo-adam-star':   {'color': '#648381', 'markevery': 4, 'zorder': 12},
        'prox-sps':         {'color': '#97BF88', 'markevery': 7, 'zorder': 6},
        'adabelief':        {'color': '#FFBF46', 'markevery': 10, 'zorder': 6},
        'adabound':         {'color': '#4f9d69', 'markevery': 10, 'zorder': 5},
        'lion':             {'color': '#dbabab', 'markevery': 10, 'zorder': 4},
        'muon':             {'color': '#2E86AB', 'markevery': 7, 'zorder': 3},
        'muon-gd':          {'color': '#6A4C93', 'markevery': 6, 'zorder': 2},
        'muon-gd-stale':    {'color': '#1B998B', 'markevery': 5, 'zorder': 4},
        'muon-l2':          {'color': '#F4D35E', 'markevery': 5, 'zorder': 2},
        'muon-gd-l2':       {'color': '#E26D5A', 'markevery': 6, 'zorder': 3},
        'default':          {'color': 'grey','markevery': 3, 'zorder': 1},
}

# more colors:
#7fb285

ALL_MARKER = ('o', 'v', 'H', 's', '>', '<' , '^', 'D', 'x')
nan_mean_fun = lambda x: x.mean(skipna=False)

class Record:
    def __init__(self, 
                 exp_id: Union[str, list], 
                 output_dir: str=DEFAULTS.output_dir, 
                 as_json: bool=True
                 ):
        
        self.exp_id = exp_id
        self.aes = copy.deepcopy(AES)

        # exp_id can be str or list (if we want to merge multiple output files)
        if isinstance(exp_id, str):
            exp_id = [exp_id]
        else:
            warnings.warn("Loading from multiple output files. Contents will be merged.")   
        
        # TODO: make this safer
        self.exp_id_str = exp_id[0]
        print(f"Creating Record with ID {self.exp_id_str}.")
        
        # object to store everything
        self.data = list()

        for _e in exp_id:
            C = Container(name=_e, output_dir=output_dir, as_json=as_json)
            print(f"Loading data from {output_dir+_e}")
            C.load() # load data

            self.data += C.data # append

        self.raw_df = self._build_raw_df()
        self.id_df = self._build_id_df()
        self.base_df = self._build_base_df(agg='mean')
        return
    
    def filter(self, drop=dict(), keep=dict(), any=False):
        """Filter out by columns in id_df. 
        
        if all=True:
         * Drops if any condition is true.
         * Keeps if all conditions are true.
        if all=False:
         * Drops if all conditions are true.
         * Keeps if any conditions is true.        

        For example, use drop={'name': 'adam'} to drop all results from Adam. 
        
        NOTE: This overwrites the base_df and id_df object.
        """
        all_ix = list()

        for k, v in drop.items():
            if not isinstance(v, list):
                v = [v] # if single value is given convert to list
            
            ix = ~self.id_df[k].isin(v) # indices to drop --> negate
            all_ix.append(ix)

        for k, v in keep.items():
            if not isinstance(v, list):
                v = [v] # if single value is given convert to list
            
            ix = self.id_df[k].isin(v) # indices to keep
            all_ix.append(ix)

        if not any:
            ixx = pd.concat(all_ix, axis=1).all(axis=1) # keep where all True
        else:
            ixx = pd.concat(all_ix, axis=1).any(axis=1) # keep where any True
            
        ids_to_keep = ixx.index[ixx.values]

        self.base_df = self.base_df[self.base_df.id.isin(ids_to_keep)]
        self.id_df = self.id_df.loc[ids_to_keep]

        warnings.warn("filter method overwrites base_df and id_df.")

        return 

    def _build_raw_df(self):
        """ create DataFrame with the stored output. Creates an id column based on opt config. """
        df_list = list()
        for r in self.data:
            this_df = pd.DataFrame(r['history'])
            
            opt_dict = copy.deepcopy(r['config']['opt'])
            opt_dict = {'name': opt_dict.pop('name'), **opt_dict} # move name to beginning
            
            id = list()
            for k, v in opt_dict.items():       
                id.append(k+'='+ str(v)) 
                
            this_df['id'] = ':'.join(id) # identifier given by all opt specifications
            this_df['run_id'] = r['config']['run_id'] # differentiating multiple runs

            # convert step-wise logs to series
            # pandas has a bug that we cannot insert a series into this_df.loc[row_ix, ..]
            # but we can insert into row, so we concat the modified rows
            if 'step_logs' in this_df.columns:
                HAVE_STEPWISE_LR = bool(opt_dict.get("stepwise_schedule"))

                # some sanity checks/warnings
                if HAVE_STEPWISE_LR and not ("stepwise_lr" not in this_df.columns):
                    warnings.warn(f"ID {id}: Expected stepwise LR log, but none found. This might cause errors in analysis.")
                if not HAVE_STEPWISE_LR and ("stepwise_lr" in this_df.columns):
                    warnings.warn(f"ID {id}: Found stepwise LR log, but no stepwise LR scheduling. Will be overwritten.")
                    
                new_rows = list()
                for row_ix, row in this_df.iterrows():
                    for k, v in row['step_logs'].items():
                        row['stepwise_' + k] = pd.Series(v, name=k)
                    
                    # reconstruct when no stepwise LR scheduling
                    if not HAVE_STEPWISE_LR:
                        # use step indices from last item, and ffill the learning rate from this epoch
                        _reconstructed_lr = dict(zip(v.keys(), row['learning_rate'] * np.ones(len(v))))
                        row['stepwise_lr'] = pd.Series(_reconstructed_lr, name='lr')
                
                    new_rows.append(row)
                
                this_df = pd.DataFrame(new_rows)
                this_df = this_df.drop(columns=['step_logs'])

            df_list.append(this_df)
            
        df = pd.concat(df_list)
        df.insert(0, 'id', df.pop('id')) # move id column to front
        df = df.reset_index(drop=True)
        
        # raise error if duplicates
        if df.duplicated(subset=['id', 'epoch', 'run_id']).any():
            raise KeyError("There seem to be duplicates (by id, epoch, run_id). Please check the output data.")

        return df
    
    def _build_id_df(self):
        """ create a DataFrame where each id is split up into all hyperparameter settings """
        id_cols = list()
        all_ids = self.raw_df.id.unique()
        for i in all_ids:
            d = id_to_dict(i)
            id_cols.append(d)
        
        id_df = pd.DataFrame(id_cols, index=all_ids)
        id_df.fillna('none', inplace=True)
        return id_df
        
    
    def _build_base_df(self, agg='mean'):
        raw_df = self.raw_df.copy()
        
        # compute mean for each id and(!) epoch
        if agg == 'mean':
            # Create an aggregation dictionary first
            # 1) if column numeric, take mean
            # 2) for stepwise logs, each element is a series --> use custom function
            # 3) else take first (e.g. for step_size_list)
            agg_dict = dict()
            for c in raw_df.columns:
                if c in ["id", "epoch"]:
                    continue
                if is_numeric_dtype(raw_df[c]):
                    agg_dict[c] = nan_mean_fun
                elif isinstance(raw_df[c][0], pd.Series):
                    agg_dict[c] = average_series_and_wrap
                else:
                    agg_dict[c] = "first"
        
            self._base_df_agg_dict = agg_dict

            df = raw_df.groupby(['id', 'epoch'], sort=False).agg(agg_dict).drop('run_id',axis=1)
            
            # only compute std for float columns
            # std returns nan if some entrys are nan
            std_columns = [c for c in raw_df.columns if is_numeric_dtype(raw_df[c])]
            df2 = raw_df.groupby(['id', 'epoch'], sort=False)[std_columns].std().drop('run_id',axis=1)           
            df2.columns = [c+'_std' for c in df2.columns]
            
            df = pd.concat([df,df2], axis=1)
            df = df.reset_index(level=-1) # moves epoch out of index
            
        elif agg == 'first':
            df = raw_df.sort_values(['id', 'epoch', 'run_id']).groupby(['id', 'epoch'], sort=False).first()
            assert len(df.run_id.unique()) == 1
            df = df.drop('run_id', axis=1)
            df = df.reset_index(level=-1) # moves epoch out of index
        
        df = df.reset_index(drop=False) # set index as integers
        df = df.merge(self.id_df, how='left', left_on='id', right_index=True) # add columns from id_df
        
        return df
    
    def build_sweep_df(self,
                       score: str='val_score',
                       xaxis: str='lr',
                       ignore_columns: list=list(),
                       cutoff: Optional[Tuple]=None
        ):

        base_df = self.base_df.copy()
        id_df = self.id_df.copy()

        grouped = base_df.groupby(['name', xaxis])
        max_epoch = grouped['epoch'].max()
        assert len(max_epoch.unique()) == 1, "It seems that different setups ran for different number of epochs."

        if cutoff is None:
            cutoff_epoch = (max_epoch[0], max_epoch[0])
        else:
            assert len(cutoff) == 2, f"Cutoff needs to be tuple (len 2), but is given as {cutoff}."
            cutoff_epoch = (cutoff[0], cutoff[1])

        # filter epochs
        sub_df = base_df[(base_df.epoch >= cutoff_epoch[0])
                         &
                         (base_df.epoch <= cutoff_epoch[1])] 
        # select the columns to group by
        grouping_cols = [c for c in id_df.columns if c not in ignore_columns]
        # group by all id_cols
        df = sub_df.groupby(grouping_cols)[[score, score+'_std']].mean()
        # move xaxis out of grouping
        df = df.reset_index(level=xaxis)
        # make xaxis float
        df[xaxis] = df[xaxis].astype('float')
        
        # get method and learning rate with best score
        # best_ind, best_x = df.index[df[s].argmax()], df[xaxis][df[s].argmax()]

        return df
    
    #============ DATABASE =================================
    #=======================================================

    def to_csv(self, name: str, df: pd.DataFrame=None, db_dir: str='stepback/records/'):
        """Create a Record csv for an experiment.

        Parameters
        ----------
        name : str
            name of csv file, recommended to be identical to ```exp_id```
        df : pd.DataFrame, optional
            Dataframe to store. By default uses ```self.base_df```.
        db_dir : str, optional
            directory for storing the csv, by default 'output/records/'.

        """
        
        if df is None:
            db = self.base_df.copy()
        else:
            db = df.copy()
        
        # sort
        db = db.sort_values(['id', 'epoch'])

        # train time depends on hardware and is not meaningful
        if 'train_epoch_time' in db.columns:
            db = db.drop(columns=['train_epoch_time', 'train_epoch_time_std'])

        db.to_csv(db_dir+name+'.csv', index=False)

        return

    #============ PLOTTING =================================
    #=======================================================
    def _reset_marker_cycle(self):
        for m in self.aes.keys():
            self.aes[m]['marker_cycle'] = itertools.cycle(ALL_MARKER)  
        return
    
    def plot_metric(self, s, df=None, log_scale=False, ylim=None, legend=True, figsize=(4,4), ax=None):
        
        if df is None:
            df = self.base_df.copy()
        
        # has to be set freshly every time
        self._reset_marker_cycle()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        names_legend = list()
        if legend:
            alpha = 1
            markersize = 6
        else:
            alpha = .65
            markersize = 4
            
            
        for m in df.id.unique():
            this_df = df[df.id==m]
            x = this_df.loc[:,'epoch']
            y = this_df.loc[:,s]
            conf = id_to_dict(m) 
            
            # construct label
            label = conf['name'] + ', ' + r'$\alpha_0=$' + conf['lr']
            for k,v in conf.items():
                if k in ['name', 'lr']:
                    continue
                label += ', ' + k + '=' + str(v)
            
            # plot
            if not y.isna().all():
                names_legend.append(conf['name'])
                ax.plot(x, 
                        y, 
                        c=self.aes.get(conf['name'], self.aes['default']).get('color'), 
                        marker=next(self.aes.get(conf['name'], self.aes['default']).get('marker_cycle')) if legend else 'o', 
                        markersize=markersize, 
                        markevery=(self.aes.get(conf['name'], self.aes['default']).get('markevery'), 20), 
                        alpha = alpha,
                        label=label,
                        zorder=self.aes.get(conf['name'], self.aes['default']).get('zorder')
                        )
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(SCORE_NAMES.get(s, s))
        ax.grid(which='both', lw=0.2, ls='--', zorder=-10)
        
        if log_scale:
            ax.set_yscale('log')    
        if ylim:
            ax.set_ylim(ylim)
        
        # full legend or only solver names
        if legend:
            ax.legend(fontsize=8, loc='lower left').set_zorder(100)
        else:
            names_legend = set(names_legend)
            handles = [Line2D([0], [0], color=self.aes.get(n, self.aes['default']).get('color'), lw=4) for n in names_legend]
            ax.legend(handles, names_legend, loc='lower left').set_zorder(100)

        fig.tight_layout()
        return fig, ax

def average_series_by_index(series_of_series):
    """
    Aggregates a Series of Series by averaging values at overlapping indices.

    Args:
        series_of_series: A pandas Series where each element is itself a pandas Series.
                          This is what a groupby().agg() operation passes to the function.
    Returns:
        A single pandas Series with the aggregated and averaged values.
    """
    concatenated_series = pd.concat(list(series_of_series))
    avg_series = concatenated_series.groupby(concatenated_series.index).mean()
    return avg_series

def average_series_and_wrap(series_of_series):
    """
    This is on ly a trick as .agg() insert a list, and not a Series.
    Thanks to Gemini.
    """
    result_series = average_series_by_index(series_of_series)
    return [result_series]

def id_to_dict(id):
    """utility for creating a dictionary from the identifier"""
    tmp = id.split(':')
    d = dict([j.split('=') for j in tmp])
    return d


def create_label(id, subset=None):
    d = id_to_dict(id)
    if subset is None:
        s = [key_to_math(k) +'='+ v for k,v in d.items()]
    else:
        s = [key_to_math(k) +'='+ v for k,v in d.items() if k in subset]
        
    return ', '.join(s)

def key_to_math(k):
    """translates column names from id_dict to math symbol"""
    if k == 'lr':
        k2 = r'$\alpha_0$'
    elif k == 'beta':
        k2 = r'$\beta$'
    elif k == 'weight_decay':
        k2 = r'$\lambda$'
    return k2
