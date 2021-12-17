import pandas as pd
import numpy as np
import argparse


parser = argparse.ArgumentParser(description="")
parser.add_argument('-expr_dir', required=True, default=None)
parser.add_argument('-positive_path', required=True, default=None)
parser.add_argument('-negative_path', required=True, default=None)
parser.add_argument('-func_name', required=True, default=None)
parser.add_argument('-top_n', type=int, default=10)

args = parser.parse_args()


class neighbor_analysis():
    def __init__(self, expr_dir, positive_path, negative_path):
        self.expr_path = expr_dir
        self.timepoints = 3
        self.whole_RPKM = None
        self.gene_map = None
        self.cov_matrix = None
        self.positive_path = positive_path
        self.negative_path = negative_path
        
        
        ## initialization and loading rpkm data
        _, _, self.whole_RPKM , self.gene_map= self.load_real_rpkm_data(expr_dir,3)
  
    def load_real_rpkm_data(self, filedir, n_timepoints):
        if not filedir == 'None' and int(n_timepoints):
            sample_size_list = []
            total_RPKM_list = []
            sample_sizex = []
            for indexy in range (0,int(n_timepoints)): # time points number
                store = pd.HDFStore(filedir+'/'+'RPKM_'+str(indexy)+'.h5') #    # scRNA-seq expression data                        )#
                rpkm = store['RPKM']            
                store.close()
                total_RPKM_list.append(rpkm)
                sample_size_list = sample_size_list + [indexy for i in range (rpkm.shape[0])] #append(rpkm.shape[0])
                sample_sizex.append(rpkm.shape[0])
                samples = np.array(sample_size_list)
                
            total_RPKM = pd.concat(total_RPKM_list, ignore_index=True)
            total_RPKM.columns = total_RPKM.columns.map(lambda x:x.lower())
            gene_map = total_RPKM.columns.values
            return samples,sample_sizex,total_RPKM, gene_map

    def calculate_cov(self):
        expr = self.whole_RPKM.iloc[:][:]   
        expr=np.asarray(expr)  
        expr = expr.transpose()
        cov_matrix = np.cov(expr)
        self.cov_matrix = cov_matrix
        return cov_matrix

    def load_posi_nega_genesets(self):
        known = np.load(self.positive_path)
        known_index = []
        for i in range(len(known)):
            known_index.append(int(np.where(self.gene_map == known[i])[0]))         
        unknown = np.load(self.negative_path)
        return known, unknown, known_index
        
    def get_top_neighbor_genes(self, geneA, known_index, top_n=10):     
        np.seterr(divide='ignore',invalid='ignore') # ignore zero/zero or one/zero
                        
        index = int(np.where(self.gene_map == geneA)[0])
        cov_list_geneA = self.cov_matrix[index, :]
        cov_list_geneA = cov_list_geneA.ravel()   
        
        cov_list_geneA = np.delete(cov_list_geneA,known_index) # avoid leak
        sub_gene_map = np.delete(self.gene_map,known_index)
        
        the_order = np.argsort(-cov_list_geneA)
        select_index = the_order[0:top_n] # top_cov number
        GeneA_neighbor_list=sub_gene_map[select_index]
         
        return GeneA_neighbor_list          


    
    
if __name__ == '__main__':

    # expr_dir = '/home/comp/csyuxu/dynDeepDRIM/data/mouse_cortex' # expression data path
    # positive_path = '/home/comp/csyuxu/dynDeepDRIM/DB_func_annotation/immune_known_gene.npy'
    # negative_path = '/home/comp/csyuxu/dynDeepDRIM/DB_func_annotation/immune_unknown_gene.npy'
    # func_name = 'immune'
    # top_n = 10
    func_name = args.func_name
    top_n = args.top_n
    
   
    ins = neighbor_analysis(args.expr_dir, args.positive_path, args.negative_path)
    ins.calculate_cov()
    posi, nega, posi_idx = ins.load_posi_nega_genesets()
    
    
    ## neighbor genes of positive cases
    Gene_list = []
    for i in range(len(posi)):
        GeneName = ins.get_top_neighbor_genes(posi[i], posi_idx, top_n)     
        Gene_list.append(GeneName)
        
    f = open('./posi_'+ func_name+'_neighborname.txt','w')   
    for i in range(len(Gene_list)):
        tmp = Gene_list[i]
        for j in range(top_n):
            f.write(tmp[j]+'\t')
        f.write('\n')
    f.close()
    
    
    ## neighbor genes of negative cases
    Gene_list = []
    for i in range(len(nega)):
        GeneName = ins.get_top_neighbor_genes(nega[i], posi_idx, top_n)     
        Gene_list.append(GeneName)
        
    f = open('./nega_'+func_name+'_neighborname.txt','w')   
    for i in range(len(Gene_list)):
        tmp = Gene_list[i]
        for j in range(top_n):
            f.write(tmp[j]+'\t')
        f.write('\n')
    f.close()

#### python3 find_neigbor_genes.py -expr_dir -positive_path -negative_path -func_name immune -top_n 10
