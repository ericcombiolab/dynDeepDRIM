import pandas as pd

raw =pd.read_table('GO0030424_axon_mus.txt',header=None)
genes  = raw.iloc[:,1].values
genes = [g.lower() for g in genes]
f = open('axon.txt','w')
for g in genes:
    f.write(g+'\n')
f.close()

