import pandas as pd
import numpy as np

a = pd.read_table('mesc1_gene_pairs_400_num.txt',header=None).values.ravel()
b = open('mesc1_gene_pairs_400.txt')
c = a*2/3
d = c.astype('int')
e = open('m1_gene_pairs_400.txt','w')
f = open('m1_gene_pairs_400_num.txt','w')
for line in b:
    l1,l2,l3 = line.split()
    if l3 != '2':
        e.write(l1+'\t'+l2+'\t'+l3+'\n')
        
for i in range(len(d)):
    f.write(str(d[i])+'\n')
b.close()
e.close()
f.close()