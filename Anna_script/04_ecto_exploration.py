#!/usr/bin/env python3
import sys, os
import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
import scanpy as sc
import sc_aautils as scaa
import gseapy as gp
from gseapy import barplot, dotplot
from collections import Counter

adata = sc.read_h5ad('sc_analysis/adata_analyzed_ecto_wt.h5ad')

sc.pl.umap(adata, color = ['replicate','expt','diff_state','leiden'], s = 30)

udf = pd.DataFrame(adata.obsm['X_umap'], columns = ['u1','u2'], index = adata.obs.index)
udf['replicate'] = adata.obs['replicate']
udf['diff_state'] = adata.obs['diff_state']

fig, axs = plt.subplots(ncols = 2, figsize = (2*3*1.6, 3))
for ax, c in zip(axs, ['replicate','diff_state']):
    for cl in set(udf[c]):
        df = udf[udf[c]==cl]
        ax.scatter(df['u1'], df['u2'], label = cl, s = 10)
    ax.legend()
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('umap 1'); ax.set_ylabel('umap 2')
    ax.set_title(c)
    
# Pluripotency-associated genes: 
sc.pl.umap(adata, color = ['Pou5f1','Sox2','Klf4','Myc'], s = 30)
sc.pl.umap(adata, color = ['Pou5f1','Dppa5a', 'Nanog','Klf4'], s = 30)
# Ectoderm markers
sc.pl.umap(adata, color = ['Sox1','Pax6','Rmst'])
# mESCs express the pluripotency-associated genes Nanog, Pou5f1, Sox2, Utf1, Zfp42, Eras, Dppa5, Dnmt3l, Col18a1, Nodal and Gli2 https://pmc.ncbi.nlm.nih.gov/articles/PMC3403099/
dex = scaa.difGeneExpr(adata, ['leiden'])['leiden']

# coarse-grained clustering
sc.tl.leiden(adata, resolution = 0.3, key_added = 'leiden03')
sc.pl.umap(adata, color = 'leiden03')
dex = scaa.difGeneExpr(adata, ['leiden03'])['leiden03']
lfc_th = 1.1; pval_th = 1e-2
for cl in dex:
    dex[cl] = dex[cl][(dex[cl]['logfoldchanges']>lfc_th)&(dex[cl]['pvals_adj']<pval_th)]

# Differentiation success rate
pluri_cluster = '3'
diff_rate = pd.DataFrame({'pluripotent': Counter(adata.obs[adata.obs['leiden03']==pluri_cluster]['expt']), 'differentiated': Counter(adata.obs[adata.obs['leiden03']!=pluri_cluster]['expt'])}).T
diff_rate = diff_rate/diff_rate.sum()

# GO
def go_analisis(genelist, background, pvth = 1e-3):
    pea = gp.enrichr(gene_list = genelist, gene_sets =  ['GO_Biological_Process_2018'], background = background, organism = 'mouse')
    pea = pea.results[['Term','P-value','Adjusted P-value','Odds Ratio','Genes']]
    return pea[pea['Adjusted P-value']<pvth]

go = {cl: go_analisis(genelist = list(dex[cl].index), background = adata.raw.var.index, pvth = 1e-1) for cl in dex}


