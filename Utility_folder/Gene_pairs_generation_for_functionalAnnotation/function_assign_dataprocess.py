import pandas as pd
import numpy as np
import os
import sys
import random



#overlap -> known and unknown set
def overlap_fgene_exprgene(gene_list_path,fgenepath):
    gene_list = [i.lower() for i in gene_list_path]
    file =open(fgenepath)
    known_gene =[]
    for line in file:
        known_gene.append(line.split()[0].lower())
    file.close()
    overlap_gene = list(set(known_gene)&set(gene_list))
    
    n_known_gene = len(overlap_gene)
    for i in range(n_known_gene):
        gene_list.remove(overlap_gene[i])
    
    unknown_index = random.sample(range(0,len(gene_list)),n_known_gene)
    unknown_set = np.array(gene_list)[unknown_index].tolist()
    return overlap_gene,unknown_set


def Generate_train_genepair_label(train_known,train_unknown,n_train):
    G1=[]
    G2=[]
    label = []
    for i in range(n_train):
        GeneA = train_known[i]
        for j in range(n_train):
            GeneB = train_known[j]
            GeneC = train_unknown[j]
            G1.append(GeneA)
            G2.append(GeneB)
            label.append('1')
            G1.append(GeneA)
            G2.append(GeneC)
            label.append('0')
    return G1,G2,label
    
def Generate_test_genepair_label(train_known,test_known,test_unknown,n_train,n_test):
    G1=[]
    G2=[]
    label = []
    for i in range(n_train):
        GeneA = train_known[i]
        for j in range(n_test):
            GeneB = test_known[j]
            GeneC = test_unknown[j]
            G1.append(GeneA)
            G2.append(GeneB)
            label.append('1')
            G1.append(GeneA)
            G2.append(GeneC)
            label.append('0')
    return G1,G2,label
    
##########

# modify here
data_path = '/tmp/csyuxu/data/mouse_cortex'
function_name = 'cognition' 
save_dir = '/home/comp/csyuxu/dynDeepDRIM/DB_function_annotation'



gene_list =np.load(os.path.join(data_path,'gene_list.npy'),allow_pickle=True).tolist()  # This input file gene_list.npy is extracted from the expression dataset
known,unknown = overlap_fgene_exprgene(gene_list, function_name+'.txt')

np.save(os.path.join(save_dir, function_name+'_known_gene'+ str(n_smallpair)), known)
np.save(os.path.join(save_dir, function_name+'_unknown_gene'+ str(n_smallpair)), unknown)


# n_train = int(len(known)*2/3)
# n_test = int(len(known)-n_train)
# train_known = known[:n_train]
# test_known = known[n_train:]
# train_unknown = unknown[:n_train]
# test_unknown = unknown[n_train:]

n_train = int(len(known)*3/5)
n_val = int(len(known)*1/5)
n_test = int(len(known)-n_train-n_val)
train_known = known[:n_train]
val_known =known[n_train:n_train+n_val]
test_known = known[n_train+n_val:]
train_unknown = unknown[:n_train]
val_unknown =unknown[n_train:n_train+n_val]
test_unknown = unknown[n_train+n_val:]


G1,G2,label = Generate_train_genepair_label(train_known,train_unknown,n_train)
G1_test,G2_test,label_test = Generate_test_genepair_label(train_known,test_known,test_unknown,n_train,n_test)
G1_val,G2_val,label_val = Generate_test_genepair_label(train_known,val_known,val_unknown,n_train,n_val)


f = open(os.path.join(save_dir,function_name+'_gene_pairs_train.txt'),'w')
for i in range(len(G1)):
    f.write(G1[i]+'\t'+G2[i]+'\t'+label[i]+'\n')
f.close()

f1 = open(os.path.join(save_dir,function_name+'_gene_pairs_test.txt'),'w') 
for i in range(len(G1_test)):
    f1.write(G1_test[i]+'\t'+G2_test[i]+'\t'+label_test[i]+'\n')
f1.close()

f1 = open(os.path.join(save_dir,function_name+'_gene_pairs_val.txt'),'w') 
for i in range(len(G1_val)):
    f1.write(G1_val[i]+'\t'+G2_val[i]+'\t'+label_val[i]+'\n')
f1.close()


count_set = [0]
count = 0
for i in range(n_train):
    count += n_test*2
    count_set.append(count)
    
f2 = open(os.path.join(save_dir,function_name+'_count_set_test.txt'),'w') 
for i in range(len(count_set)):
    f2.write(str(count_set[i])+'\n')
f2.close()



