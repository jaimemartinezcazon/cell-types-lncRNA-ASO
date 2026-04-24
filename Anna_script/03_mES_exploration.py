#!/usr/bin/env python3
import sys, os
import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
import scanpy as sc
import Colors
import sc_aautils as scaa

adata = sc.read_h5ad('sc_analysis/adata_analyzed_mES.h5ad')
adata.obs['origin'] = adata.obs['plateID'].apply(lambda x: '_'.join(x.rsplit('_')[:-1]))

# Pluripotency-associated genes: 
sc.pl.umap(adata, color = ['Pou5f1','Sox2','Klf4','Myc'])
# mESCs express the pluripotency-associated genes Nanog, Pou5f1, Sox2, Utf1, Zfp42, Eras, Dppa5, Dnmt3l, Col18a1, Nodal and Gli2 https://pmc.ncbi.nlm.nih.gov/articles/PMC3403099/
dex = scaa.difGeneExpr(adata, ['leiden'])['leiden']
# Wnt7a => https://pubmed.ncbi.nlm.nih.gov/34562599/
# run GO


# Clear batch effects are observed:
sc.pl.umap(adata, color = ['origin','leiden'])
cdf = pd.crosstab(adata.obs['origin'], adata.obs['leiden'])
fcdf = 100*cdf/cdf.sum()
Z, dg = scaa.hierarchicalClustering(fcdf.T)
fcdf = fcdf[fcdf.columns[dg['leaves']]]
fig, ax = plt.subplots()
b = np.zeros(len(fcdf.columns))
for idx in fcdf.index:
    ax.bar(range(len(fcdf.columns)), fcdf.loc[idx], bottom = b, label = idx)
    b += fcdf.loc[idx]
ax.set_xticks(range(len(fcdf.columns)))
ax.set_xticklabels(fcdf.columns)
ax.legend(loc = 2, bbox_to_anchor = (1,1))


udf = pd.DataFrame(adata.obsm['X_umap'], columns = ['u1','u2'], index = adata.obs.index)
