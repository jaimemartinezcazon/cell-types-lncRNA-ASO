#!/usr/bin/env python3
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv

coutb = read_csv('mES_2diffs_coutb_featurecounts_table.txt.gz', sep = '\t', comment='#', index_col = 0)
coutc = read_csv('mES_2diffs_coutc_featurecounts_table.txt.gz', sep = '\t', comment='#', index_col = 0)

def select_columns(df):
    df = df[df.columns[5:]]
    df.columns = [c.rsplit('/')[-1].rsplit('_cbc')[0] for c in df.columns]
    return df

coutb = select_columns(coutb)
coutc = select_columns(coutc)

cell_mdf = pd.DataFrame({
    'counts': coutc.sum(),
    'umis': coutb.sum(), 
    'genes': (coutb>0).sum(),
    'well': [c.rsplit('_')[-1] for c in coutb.columns],
    'replicate': [c.rsplit('_')[-2] for c in coutb.columns],
    'expt': [c.rsplit('_plate')[0].rsplit('mES_')[-1]  for c in coutb.columns],
    'ercc': coutb.loc[[idx for idx in coutb.index if 'ERCC-' in idx]].sum()
    })
cell_mdf['umis'] = cell_mdf['umis'] - cell_mdf['ercc']

cell_mdf['plate'] = cell_mdf.apply(lambda x: '_'.join(x[['expt','replicate']]), axis = 1)

def histo_cell_metadata_by_plate(col = 'umis', bins = 100, log = True):
    N = len(set(cell_mdf['plate']))
    fig, axs = plt.subplots(nrows = N, figsize = (3*1.6, 3*N))
    for ax, pl in zip(axs, sorted(set(cell_mdf['plate']))):
        df = cell_mdf[cell_mdf['plate']==pl]
        if log: 
            ax.hist(np.log10(df[col]), bins = bins, label = pl)
            ax.set_xlim( 0.9*np.log10(cell_mdf[col]).min(), np.log10(cell_mdf[col]).max()*1.1)
        else:
            ax.hist(df[col],  bins = bins, label = pl)
            ax.set_xlim( 0.9*(cell_mdf[col]).min(), (cell_mdf[col]).max()*1.1)
        ax.legend()
    return fig, axs

def scatter_cell_metadata_by_plate(col1, col2, log1 = True, log2 = True):
    N = len(set(cell_mdf['plate']))
    fig, axs = plt.subplots(nrows = N, figsize = (3*1.6, 3*N))
    for ax, pl in zip(axs, sorted(set(cell_mdf['plate']))):
        df = cell_mdf[cell_mdf['plate']==pl].copy()
        df['x'] = np.log10(df[col1]) if log1 else df[col1]
        df['y'] = np.log10(df[col2]) if log2 else df[col2]
        ax.scatter(df['x'], df['y'], s = 5, label = pl)
        ax.legend()
    return fig, axs

fig, axs = histo_cell_metadata_by_plate('umis', bins = 50)
fig.savefig('histo_log10umis.pdf', bbox_inches = 'tight')

cell_mdf['overseq'] = cell_mdf['counts']/cell_mdf['umis']
fig, axs = histo_cell_metadata_by_plate('overseq', bins = 10, log = False)
axs[-1].set_xlabel('oversequencing')
fig.savefig('histo_overseq.pdf', bbox_inches = 'tight')

fig, axs = scatter_cell_metadata_by_plate(col1 = 'umis', col2 = 'ercc')
axs[round(len(axs)/2)].set_ylabel('log10[ERCC]')
axs[-1].set_xlabel('log10(umis)')
fig.savefig('scatter_umiVSercc.pdf', bbox_inches = 'tight')

def plate_plot(df, ax, col = 'umis', log = True, scale = 20, th = 1e3):
    letter2y = {'A':15,'B':14,'C':13,'D':12,'E':11,'F':10,'G':9,'H':8,'I':7,'J':6,'K':5,'L':4,'M':3,'N':2,'O':1,'P':0}
    df['y'] = df['well'].apply(lambda x: letter2y[x[0]])
    df['x'] = df['well'].apply(lambda x: int(x[1:]))
    df['s'] = np.log10(df[col]) if log else df[col]
    df['s'] = scale*df['s']/df['s'].max()
    sdf = df[df[col]>th]
    ax.scatter(sdf['x'], sdf['y'], s = sdf['s'])
    sdf = df[df[col]<=th]
    ax.scatter(sdf['x'], sdf['y'], s = sdf['s'])
    ax.set_yticks(range(16)); ax.set_yticklabels(list(letter2y.keys())[::-1], fontsize = 8)
    ax.set_xticks(range(1,25)); ax.set_xticklabels(range(1,25), fontsize = 7)
    return ax

def plateplot_cell_metadata_by_plate(col = 'umis', log = True, scale = 20, th = 1e3):
    N = len(set(cell_mdf['plate']))
    fig, axs = plt.subplots(nrows = N, figsize = (3*1.6, 3*N))
    for ax, pl in zip(axs, sorted(set(cell_mdf['plate']))):
        df = cell_mdf[cell_mdf['plate']==pl].copy()
        plate_plot(df, ax, col = col, log = log, scale = scale, th = th)
        ax.set_title(pl)
    plt.tight_layout()
    return fig, axs

fig, axs = plateplot_cell_metadata_by_plate(col = 'umis', log = False, th = 1e3, scale = 40)
fig.savefig('plates_umi.pdf', bbox_inches = 'tight')

fig, axs = scatter_cell_metadata_by_plate(col1 = 'umis', col2 = 'genes', log1 = True, log2 = True)
axs[round(len(axs)/2)].set_ylabel('log10[genes]'); 
axs[-1].set_xlabel('log10[umis]')
fig.savefig('scatter_umiVSgenes.pdf', bbox_inches = 'tight')

fig, ax = plt.subplots()
for pl in sorted(set(cell_mdf['plate'])):
    df = cell_mdf[cell_mdf['plate']==pl].copy()
    ax.scatter(np.log10(df['umis']), np.log10(df['genes']), s = 5, label = pl)
ax.legend(loc = 2, bbox_to_anchor = (1,1), markerscale = 5)
ax.set_xlabel('log10(umis)'); ax.set_ylabel('log10(genes)')
fig.savefig('scatter_umiVSgenes_merged.pdf', bbox_inches = 'tight')


cell_mdf.to_csv('cell_metadata.tsv', sep = '\t')

fcoutb = coutb[cell_mdf.index]
fcoutc = coutc[cell_mdf.index]

gene_mdf = pd.DataFrame({
    'counts': fcoutc.sum(axis=1),
    'umis': fcoutb.sum(axis=1),
    'cells': (fcoutb>0).sum(axis=1)
    })

fcoutb = fcoutb.loc[gene_mdf[gene_mdf['cells']>0].index]
#fcoutb.to_csv('mES_2diffs_coutb_featurecounts_filtered_table.txt', sep = '\t')
# SAVE AS AN MTX!

from scipy.sparse import csr_matrix
from scipy import io
os.system('mkdir -p mES_2diffs_coutb_featurecounts_filtered_table')
mtx = csr_matrix(fcoutb.astype(pd.SparseDtype("float",0)).sparse.to_coo())
io.mmwrite("mES_2diffs_coutb_featurecounts_filtered_table/matrix.mtx", mtx)
pd.Series(fcoutb.columns).to_csv('mES_2diffs_coutb_featurecounts_filtered_table/barcodes.tsv', index = None)
pd.Series(fcoutb.index).to_csv('mES_2diffs_coutb_featurecounts_filtered_table/features.tsv', index = None)
os.system('gzip mES_2diffs_coutb_featurecounts_filtered_table/*')


