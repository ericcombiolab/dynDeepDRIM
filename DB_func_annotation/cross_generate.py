import pandas as pd
import numpy as np


cogn_genes= np.load('cognition_known_gene.npy')
axon_genes= np.load('axon_known_gene.npy')
sensory_genes= np.load('sensory_perception_known_gene.npy')
syn_genes= np.load('synaptic_transmission_known_gene.npy')


#tmp= list(set(axon_genes)-set(syn_genes)-set(cogn_genes)-set(sensory_genes))
#tmp= list(set(syn_genes)-set(axon_genes)-set(cogn_genes)-set(sensory_genes))
#tmp= list(set(cogn_genes)-set(syn_genes)-set(axon_genes)-set(sensory_genes))
tmp= list(set(sensory_genes)-set(syn_genes)-set(cogn_genes)-set(axon_genes))

#tobe_pred_genes = tmp[:10]
tobe_pred_genes = tmp


genes_type = 'sensory'



n= int(len(axon_genes)*3/5)
train_genes = axon_genes[:n]

f = open(genes_type+'_genes_test_axon.txt','w')
for g1 in train_genes:
    for g2 in tobe_pred_genes:
        f.write(g1+'\t'+g2+'\t'+str(1)+'\n')
f.close()


# n= int(len(sensory_genes)*3/5)
# train_genes = sensory_genes[:n]

# f = open(genes_type+'_genes_test_sensory.txt','w')
# for g1 in train_genes:
    # for g2 in tobe_pred_genes:
        # f.write(g1+'\t'+g2+'\t'+str(1)+'\n')
# f.close()


n= int(len(cogn_genes)*3/5)
train_genes = cogn_genes[:n]

f = open(genes_type+'_genes_test_cognition.txt','w')
for g1 in train_genes:
    for g2 in tobe_pred_genes:
        f.write(g1+'\t'+g2+'\t'+str(1)+'\n')
f.close()


n= int(len(syn_genes)*3/5)
train_genes = syn_genes[:n]

f = open(genes_type+'_genes_test_synaptic.txt','w')
for g1 in train_genes:
    for g2 in tobe_pred_genes:
        f.write(g1+'\t'+g2+'\t'+str(1)+'\n')
f.close()


