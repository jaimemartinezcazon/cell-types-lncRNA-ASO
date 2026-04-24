#!/usr/bin/env python3
import sys, os
import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
import scanpy as sc
import sc_aautils as scaa
import PyComplexHeatmap as pch
import Colors
import itertools as it

# read data
adata = sc.read_mtx('mES_2diffs_coutb_featurecounts_filtered_table/matrix.mtx.gz')
adata = adata.T
adata.obs.index = list(read_csv('mES_2diffs_coutb_featurecounts_filtered_table/barcodes.tsv.gz')['0'])
adata.var.index = list(read_csv('mES_2diffs_coutb_featurecounts_filtered_table/features.tsv.gz')['Geneid'])

# cell experimental metadata
adata.obs['well'] = [idx.rsplit('_')[-1] for idx in adata.obs.index]
adata.obs['replicate'] = [idx.rsplit('_')[-2] for idx in adata.obs.index]
adata.obs['expt'] = ['_'.join(idx.rsplit('_')[1:][:-2]) for idx in adata.obs.index]
adata.obs['plateID'] = adata.obs.apply(lambda x: '_'.join(x[['expt','replicate']]), axis = 1)
adata.obs['perturb'] = adata.obs['expt'].apply(lambda x: 'ASO' if 'aso' in x else 'WT')

# gene annotation metadata
gene_annotation = read_csv('gene_biotypes.tsv', sep = '\t')
gene_annotation = gene_annotation.groupby('gene_name').agg(lambda x: '-'.join(set(x)))
adata.var = gene_annotation.loc[adata.var.index]

# cell sequencing metadata
cell_mdf = read_csv('cell_metadata.tsv', sep = '\t')
adata.obs['ercc_counts'] = adata[:,adata.var['gene_biotype'] == 'ERCC'].X.sum(axis=1)
adata.obs['protein_coding_counts'] = adata[:,adata.var['gene_biotype'] == 'protein_coding'].X.sum(axis=1)
adata.obs['total_counts'] = adata.X.sum(axis=1)
adata.obs['tx_counts'] = adata.obs['total_counts']-adata.obs['ercc_counts']
adata = adata[adata.obs['total_counts']-adata.obs['ercc_counts']>1e3,:] # filter cells with less than 1000 useful umis (not ERCC)


# https://www.ensembl.org/info/genome/genebuild/biotypes.html
small_biotypes = ['piRNA','snRNA','snoRNA','pre_miRNA','misc_RNA','scaRNA','Mt_tRNA']
adata.var['small_biotype'] = adata.var['gene_biotype'].apply(lambda x: x in small_biotypes)
adata.obs['small_counts'] = adata[:,adata.var['small_biotype']].X.sum(axis=1)

lnc_biotypes = ['lincRNA','antisense','processed_transcript','sense_intronic','bidirectional_promoter_lncRNA','sense_overlapping','3prime_overlapping_ncRNA']
adata.var['lnc_biotype'] = adata.var['gene_biotype'].apply(lambda x: x in lnc_biotypes)
adata.obs['lnc_counts'] = adata[:,adata.var['lnc_biotype']].X.sum(axis=1)

pseudo_biotypes = ['processed_pseudogene','unprocessed_pseudogene','transcribed_processed_pseudogene','transcribed_unprocessed_pseudogene','polymorphic_pseudogene','unitary_pseudogene','pseudogene','transcribed_unitary_pseudogene','translated_unprocessed_pseudogene']
adata.var['pseudo_biotypes'] = adata.var['gene_biotype'].apply(lambda x: x in pseudo_biotypes)
adata.obs['pseudo_counts'] = adata[:,adata.var['pseudo_biotypes']].X.sum(axis=1)

df = pd.DataFrame({
    'ERCC_perc': 100*adata.obs['ercc_counts']/adata.obs['total_counts'],
    'prot_cod_perc': 100*adata.obs['protein_coding_counts']/adata.obs['total_counts'],
    'small_perc': 100*adata.obs['small_counts']/adata.obs['total_counts'],
    'lnc_perc': 100*adata.obs['lnc_counts']/adata.obs['total_counts'],
    'pseudo_perc': 100*adata.obs['pseudo_counts']/adata.obs['total_counts']
    })

Z, dg = scaa.hierarchicalClustering(df)
df = df.loc[df.index[dg['leaves']]]

expt2color = {exp: Colors.colors[i+5] for i, exp in enumerate(set(adata.obs['expt']))}
rep2color = {rep: Colors.colors[i+15] for i, rep in enumerate(set(adata.obs['replicate']))}

fig, ax = plt.subplots(figsize = (1.5*2*3*1.6, 1.5*3))
col_ha = pch.HeatmapAnnotation(
        Expt= pch.anno_simple(adata.obs.loc[df.index,'expt'], colors = expt2color, rasterized = True),
        Perturb = pch.anno_simple(adata.obs.loc[df.index,'perturb'], colors = {'ASO': 'gray', 'WT': 'white'}, rasterized = True),
        Replicate = pch.anno_simple(adata.obs.loc[df.index,'replicate'], colors = rep2color, rasterized = True), 
        Percentages = pch.anno_barplot(df, grid = False, edgecolor = None, height = 30), 
        axis = 1, plot = True, legend = True, legend_gap = 3)
plt.tight_layout()
plt.show()
plt.savefig('biotype_percentage.pdf', bbox_inches = 'tight')

df = pd.DataFrame({
    'prot_cod_perc': 100*adata.obs['protein_coding_counts']/adata.obs['tx_counts'],
    'small_perc': 100*adata.obs['small_counts']/adata.obs['tx_counts'],
    'lnc_perc': 100*adata.obs['lnc_counts']/adata.obs['tx_counts'],
    'pseudo_perc': 100*adata.obs['pseudo_counts']/adata.obs['tx_counts']
    })
Z, dg = scaa.hierarchicalClustering(df)
df = df.loc[df.index[dg['leaves']]]

fig, ax = plt.subplots(figsize = (1.5*2*3*1.6, 1.5*3))
col_ha = pch.HeatmapAnnotation(
        Expt= pch.anno_simple(adata.obs.loc[df.index,'expt'], colors = expt2color, rasterized = True),
        Perturb = pch.anno_simple(adata.obs.loc[df.index,'perturb'], colors = {'ASO': 'gray', 'WT': 'white'}, rasterized = True),
        Replicate = pch.anno_simple(adata.obs.loc[df.index,'replicate'], colors = rep2color, rasterized = True),
        Percentages = pch.anno_barplot(df, grid = False, edgecolor = None, height = 30),
        axis = 1, plot = True, legend = True, legend_gap = 3)
plt.tight_layout()
plt.show()
plt.savefig('biotype_noERCC_percentage.pdf', bbox_inches = 'tight')

df = pd.DataFrame({
    'prot_cod_perc': 100*adata.obs['protein_coding_counts']/(adata.obs['tx_counts']-adata.obs['small_counts']),
    'lnc_perc': 100*adata.obs['lnc_counts']/(adata.obs['tx_counts']-adata.obs['small_counts']),
    'pseudo_perc': 100*adata.obs['pseudo_counts']/(adata.obs['tx_counts']-adata.obs['small_counts'])
    })
Z, dg = scaa.hierarchicalClustering(df)
df = df.loc[df.index[dg['leaves']]]

fig, ax = plt.subplots(figsize = (1.5*2*3*1.6, 1.5*3))
col_ha = pch.HeatmapAnnotation(
        Expt= pch.anno_simple(adata.obs.loc[df.index,'expt'], colors = expt2color, rasterized = True),
        Perturb = pch.anno_simple(adata.obs.loc[df.index,'perturb'], colors = {'ASO': 'gray', 'WT': 'white'}, rasterized = True),
        Replicate = pch.anno_simple(adata.obs.loc[df.index,'replicate'], colors = rep2color, rasterized = True),
        Percentages = pch.anno_barplot(df, grid = False, edgecolor = None, height = 30),
        axis = 1, plot = True, legend = True, legend_gap = 3)
plt.tight_layout()
plt.show()
plt.savefig('biotype_small_percentage.pdf', bbox_inches = 'tight')

###
# sc analysis: take only protein coding and long non coding fraction...
###
out = 'sc_analysis/'
adata.write_h5ad(out + '/raw_adata.h5ad')

adata = sc.read_h5ad(out + '/raw_adata.h5ad')

stem_cell_wells = [''.join(x) for x in it.product(['A','B','C','D','E','F','G','H'],['01','02','03'])] + [''.join(x) for x in it.product(['I','J','K','L','M','N','O','P'],['01','02'])]

adata.obs['expt'] = ['mES' if adata.obs.loc[idx,'well'] in stem_cell_wells else adata.obs.loc[idx,'expt'] for idx in adata.obs.index]

adata = adata[adata.obs['protein_coding_counts']>500,:]
adata = adata[:,[adata.var.loc[idx,'small_biotype']==False for idx in adata.var.index]]
adata = adata[:,adata.var['gene_biotype']!='ERCC']
adata.obs['counts'] = adata.X.sum(axis=1)
adata.obs['genes'] = (adata.X>0).sum(axis=1)

histone_genes = [idx for idx in adata.var.index if idx[0:3] in ['H1f','H2a','H2b','H3c','H3f','H4c','H4f']]
adata.var['histone'] = [idx in histone_genes for idx in adata.var.index]
adata.obs['histone_counts'] =  adata[:, adata.var['histone']].X.toarray().sum(axis=1)
adata.obs['histone_fraction'] = adata.obs['histone_counts']/adata.obs['total_counts']
adata.obs['histone_logfraction'] = np.log(adata[:, adata.var['histone']].X.toarray()+1).sum(axis=1)/np.log(adata.X.toarray()+1).sum(axis=1)

mt_genes = [idx for idx in adata.var.index if 'mt-' in idx]
adata.var['mito_gene'] = [idx in mt_genes for idx in adata.var.index]
adata.obs['mito_counts'] = adata[:, adata.var['mito_gene']].X.toarray().sum(axis=1)
adata.obs['mito_fraction'] = adata.obs['mito_counts']/adata.obs['total_counts']

N = len(set(adata.obs['plateID']))
fig, axs = plt.subplots(nrows = N, figsize = (3*1.6, 3*N))
for p, ax in zip(sorted(set(adata.obs['plateID'])), axs):
    ax.hist(np.log10(adata.obs[adata.obs['plateID']==p]['counts']), bins = 50, label = p)
    ax.legend(); ax.set_ylabel('frequency');
    ax.set_xlim(np.log10(adata.obs['counts'].min()*0.9),np.log10(adata.obs['counts'].max()))
ax.set_xlabel('log10(counts)')
fig.savefig(out + '/histo_countsXplate.pdf', bbox_inches = 'tight')

fig, ax = plt.subplots(figsize = (3*2*1.6, 3))
for p in sorted(set(adata.obs['plateID'])):
    df = adata.obs[adata.obs['plateID']==p]
    ax.scatter(df['counts'], df['genes'], label = p, s = 5)
ax.legend(loc = 2, bbox_to_anchor = (1,1))
ax.set_xlabel('counts'); ax.set_ylabel('genes')
fig.savefig(out + '/scatter_countsVSgene_Xplate.pdf', bbox_inches = 'tight')

adata = adata[adata.obs['counts']<300000,:]

# cell cycle: s-phase
fig, axs = plt.subplots(nrows = N, figsize = (3*1.6, 3*N))
for p, ax in zip(sorted(set(adata.obs['plateID'])), axs):
    ax.hist(adata.obs[adata.obs['plateID']==p]['histone_fraction'], bins = 50, label = p)
    ax.legend(); ax.set_ylabel('frequency');
    ax.set_xlim(adata.obs['histone_fraction'].min()*0.9,adata.obs['histone_fraction'].max())
ax.set_xlabel('histone_fraction')
fig.savefig(out + '/histo_histone_fraction.pdf', bbox_inches = 'tight')

# mito genes
fig, axs = plt.subplots(nrows = N, figsize = (3*1.6, 3*N))
for p, ax in zip(sorted(set(adata.obs['plateID'])), axs):
    ax.hist(adata.obs[adata.obs['plateID']==p]['mito_fraction'], bins = 50, label = p)
    ax.legend(); ax.set_ylabel('frequency');
    ax.set_xlim(adata.obs['mito_fraction'].min()*0.9,adata.obs['mito_fraction'].max())
ax.set_xlabel('mito_fraction')
fig.savefig(out + '/histo_mito_fraction.pdf', bbox_inches = 'tight')

fig, axs = plt.subplots(nrows = N, figsize = (3*1.6, 3*N))
for p, ax in zip(sorted(set(adata.obs['plateID'])), axs):
    df = adata.obs[adata.obs['plateID']==p]
    ax.scatter(df['counts'], df['mito_fraction'], label = p)
    ax.legend(); ax.set_ylabel('mito fraction');
    ax.set_ylim(adata.obs['mito_fraction'].min()*0.9,adata.obs['mito_fraction'].max())
ax.set_xlabel('counts')
fig.savefig(out + '/scatter_mitoVScounts.pdf', bbox_inches = 'tight')

adata = adata[adata.obs['mito_fraction']<0.15,:]

# start analysis
outdir = out

adatas = {'all': adata, 
          'ecto_all': adata[['ectodiff' in adata.obs.loc[idx,'plateID'] for idx in adata.obs.index],:],
          'ecto_wt': adata[['ectodiff_plate' in adata.obs.loc[idx,'plateID'] for idx in adata.obs.index],:],
          'meso_wt': adata[['mesodiff' in adata.obs.loc[idx,'plateID'] for idx in adata.obs.index],:],
          'mES': adata[adata.obs['expt']=='mES',:]
          }

conditions = list(adatas.keys())

for c in conditions:
    sc.pp.normalize_total(adatas[c], target_sum = 1e4)
    sc.pp.log1p(adatas[c])
    adatas[c].raw = adatas[c]

for c in conditions:
    sc.pp.highly_variable_genes(adatas[c], min_mean=0.1, max_mean=3, min_disp=0.5)
    print(c, adatas[c].var.highly_variable.sum())

N = len(conditions)
fig, axs = plt.subplots(ncols = N, figsize = (2*N*1.6, 2))
for ax, c in zip(axs, conditions):
    adata = adatas[c]
    for hv in set(adata.var['highly_variable']):
        aux = adata.var[adata.var['highly_variable']==hv]
        ax.scatter(aux['means'], aux['dispersions'], s = 2, label = str(hv) + ' ('+str(len(aux)) + ')')
    ax.set_title(c);  ax.set_xlabel('mean'); ax.set_ylabel('dispersion'); ax.legend();
fig.savefig(outdir + '/scatter_highlyvariable.pdf', bbox_inches = 'tight')

for c in conditions:
    print(c)
    print(adatas[c].shape)
    adatas[c] = adatas[c][:, adatas[c].var.highly_variable]
    print(adatas[c].shape)
    adatas[c] = adatas[c][:, np.invert(adatas[c].var.histone)]
    print(adatas[c].shape)
    sc.pp.regress_out(adatas[c], ['counts'])
    sc.pp.scale(adatas[c], max_value=10)

pcadfs = {}
genecoef_pcadf = {}
rnd_vars = {}
for c in conditions:
    rnd_vars[c] = []
    for i in range(5):
        print('randomization ',str(i),'for ',c)
        v = adatas[c].X.reshape(adatas[c].shape[0]*adatas[c].shape[1])
        np.random.shuffle(v)
        v = v.reshape(adatas[c].X.shape)
        V = np.cov(v.T)
        eigval, eigvec = np.linalg.eig(V)
        rnd_vars[c].append(eigval)
    sc.tl.pca(adatas[c], svd_solver='arpack', n_comps = min([min(adatas[c].shape)-1,1000]))
    pcadfs[c] = pd.DataFrame(adatas[c].obsm['X_pca'], index = adatas[c].obs.index)
    genecoef_pcadf[c] = pd.DataFrame(adatas[c].varm['PCs'], index = adatas[c].var.index)

fig, axs = plt.subplots(ncols = N, nrows = 3, figsize = (3*N*1.6, 3*3))
for axx, c in zip(axs.T, conditions):
    ax = axx[0]
    for d in set(adatas[c].obs['expt']):
        cells = adatas[c][adatas[c].obs['expt']==d,].obs.index
        ax.scatter(pcadfs[c].loc[cells,0], pcadfs[c].loc[cells,1], s = 2, alpha = 0.5, label = d)
#    ax.legend()
    ax.set_xlabel("PCA 1 ("+"{:.2f}".format(100*adatas[c].uns['pca']['variance_ratio'][0])+"%)")
    ax.set_ylabel("PCA 2 ("+"{:.2f}".format(100*adatas[c].uns['pca']['variance_ratio'][1])+"%)")
    ax.set_title(c)
    ax = axx[1]
    for g in np.abs(genecoef_pcadf[c][0]).sort_values(ascending=False).index[:8]:
        ax.plot([0,genecoef_pcadf[c].loc[g,0]], [0,genecoef_pcadf[c].loc[g,1]])
        ax.text(genecoef_pcadf[c].loc[g,0],genecoef_pcadf[c].loc[g,1],g)
    ax.set_xticks([]); ax.set_xlabel("PC 1"); ax.set_yticks([]); ax.set_ylabel("PC 2");
    ax = axx[2]
    for g in np.abs(genecoef_pcadf[c][1]).sort_values(ascending=False).index[:8]:
        ax.plot([0,genecoef_pcadf[c].loc[g,0]], [0,genecoef_pcadf[c].loc[g,1]])
        ax.text(genecoef_pcadf[c].loc[g,0],genecoef_pcadf[c].loc[g,1],g)
    ax.set_xticks([]); ax.set_xlabel("PC 1"); ax.set_yticks([]); ax.set_ylabel("PC 2");
axs[0][-1].legend(loc = 2, bbox_to_anchor = (1,1))
fig.savefig(outdir + '/PC1vsPC2.pdf', bbox_inches = 'tight')

n_pcas = {}
fig, axs = plt.subplots(ncols = N, figsize = (3*N*1.6, 3))
ncomps = 70
for ax, c in zip(axs, conditions):
    rnd_mean = [np.mean([x[i] for x in rnd_vars[c]]) for i in range(len(rnd_vars[c][0]))]
    rnd_stderr = [np.std([x[i] for x in rnd_vars[c]])/np.sqrt(len(rnd_vars[c])-1) for i in range(len(rnd_vars[c][0]))]
    rnd_max = [np.max([x[i] for x in rnd_vars[c]]) for i in range(len(rnd_vars[c][0]))]
    var_th = rnd_mean[np.array(rnd_mean).argmax()] + rnd_stderr[np.array(rnd_stderr).argmax()]
    npca = max([(adatas[c].uns['pca']['variance']>1.2*var_th).sum(),5])
    n_pcas[c] = npca
    ax.scatter(range(ncomps), adatas[c].uns['pca']['variance_ratio'][:ncomps], s = 5, label = c)
    #ax.errorbar(np.arange(1,len(rnd_vars[c][0])+1), rnd_mean, yerr = rnd_stderr, color = 'orange', label = 'mean randomized')
    ax.text(npca, 0.5*(adatas[c].uns['pca']['variance_ratio'].max()+adatas[c].uns['pca']['variance_ratio'].min()), ' '+str(npca), ha = 'left')
    ax.legend()
    #ax.set_yscale('log')
    ax.axvline(npca, ls = '--', c = 'k');
    #ax.set_xlim(-5,min([3*npca, len(rnd_vars[0])+1])+10)
    ax.axvline(15, c = 'r', ls = '--')
fig.savefig(outdir + '/PCA_variance.pdf', bbox_inches = 'tight')

print(n_pcas)

for c in conditions:
    sc.pp.neighbors(adatas[c], n_neighbors = 10, n_pcs = n_pcas[c],  metric = 'manhattan')
    sc.tl.umap(adatas[c], n_components = 2, random_state = 235123,  min_dist = 0.1, spread = 0.75, maxiter = 500)
    sc.tl.leiden(adatas[c])

for c in conditions:
    adatas[c].obs['germ_layer'] = adatas[c].obs['plateID'].apply(lambda x: x.rsplit('diff_')[0])

for c in conditions:
    adatas[c].obs['diff_state'] = adatas[c].obs.apply(lambda x: 'mES' if x['expt']=='mES' else x['germ_layer'], axis = 1)


cols = ['expt','plateID','perturb', 'germ_layer', 'diff_state']
for col in cols:
    fig, axs = plt.subplots(ncols = N, nrows = 3, figsize = (3*N*1.6, 3*3))
    for axx, c in zip(axs.T, conditions):
        udf = pd.DataFrame(adatas[c].obsm['X_umap'], columns = ['u1','u2'], index = adatas[c].obs.index)
        udf[col] = adatas[c].obs[col];
        udf['leiden'] = adatas[c].obs['leiden'].astype(int)
        ax = axx[0]
        for d in set(udf[col]):
            sudf = udf[udf[col]==d]
            ax.scatter(sudf['u1'], sudf['u2'], label = d, s = 1)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel('umap 1'); ax.set_ylabel('umap 2')
        ax.set_title(c)
        ax = axx[1]
        for i, d in enumerate(set(udf['leiden'])):
            sudf = udf[udf['leiden']==d]
            ax.scatter(sudf['u1'], sudf['u2'], label = d, s = 1, c = Colors.colors[i])
            ax.text(sudf['u1'].mean(), sudf['u2'].mean(), d, va = 'center', ha = 'center')
            ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel('umap 1'); ax.set_ylabel('umap 2')
        ax = axx[2]
        cdf = pd.crosstab(udf['leiden'], udf[col])
        bottom = np.zeros(len(cdf))
        for ph in set(udf[col]):
            ax.bar(cdf.index, cdf[ph], label = ph, bottom = bottom)
            bottom += cdf[ph]
        ax.legend(loc = 2, bbox_to_anchor = (0,0)); ax.set_xticks(range(len(cdf))); ax.set_xticklabels(cdf.index, rotation = 90, fontsize = 8)
        ax.set_ylabel('cell number'); ax.set_xlabel('leiden clusters')
    fig.savefig(outdir + '/umap_leidenX'+col+'.pdf', bbox_inches = 'tight')

for c in conditions:
    sc.pl.umap(adatas[c], color = ['histone_fraction'])

for c in conditions:
    adatas[c].write_h5ad(out + '/adata_analyzed_'+c+'.h5ad')





