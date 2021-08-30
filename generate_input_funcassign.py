import pandas as pd
import numpy as np
import pickle
import argparse
import os

parser = argparse.ArgumentParser(description="")
parser.add_argument('-genepairs_filepath',required=True, default=None, help="gene pairs file format: gene1 gene2 label")
parser.add_argument('-expression_filepath',required=True, default=None, help=".h5 file, expression values are RPKM")
parser.add_argument('-func_geneset_filepath',required=True, default=None, help="The shared gene between download genes and the genes in expression profiles")
parser.add_argument('-n_timepoints',required=True, default=None, help="The number of time points")
parser.add_argument('-save_dir',required=True, default=None, help="The dir of output file")
parser.add_argument('-save_filename',type=str,required=True, default=None, help="The name of output file")

args = parser.parse_args()

def load_gene_pair(file_path):
    GeneA_list = []
    GeneB_list = []
    label_list = []
    s= open(file_path,'r')
    for line in s:
        GeneA, GeneB, label=line.split()
        GeneA_list.append(GeneA)
        GeneB_list.append(GeneB)
        label_list.append(label)
    return GeneA_list,GeneB_list,label_list
    
def load_real_data(filedir, n_timepoints):
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
        
def get_histogram_bins_time(RPKM, geneA, geneB, samples,sample_size):
    x_geneA=RPKM[geneA]
    x_geneB=RPKM[geneB]
    H =[]
    if x_geneA is not None:
        if x_geneB is not None:
            
            x_1 = np.array(np.log10(x_geneA + 10 ** -2) ) 
            x_2 = np.array(np.log10(x_geneB + 10 ** -2))
         
            datax = np.concatenate((x_1[:, np.newaxis], x_2[:, np.newaxis], samples[:, np.newaxis]), axis=1)           
            H, edges = np.histogramdd(datax, bins=(8, 8, 3))
                       
            HT = (np.log10(H / sample_size + 10 ** -3) + 3) / 3
            H2 = np.transpose(HT, (2, 0, 1))
            
            return H2
        else:
            return None
    else:
        return None
  
def calculate_cov(rpkm):
    expr = rpkm.iloc[:][:]   
    expr=np.asarray(expr)  
    expr = expr.transpose()
    cov_matrix = np.cov(expr)
    #corr_matrix = np.corrcoef(expr)
    return cov_matrix
    #return corr_matrix
    
def get_images_with_top_cov_pairs(geneA,geneB,cov_matrix,gene_map,whole_RPKM,sample_index,sample_size, train_known_index):
    np.seterr(divide='ignore',invalid='ignore') # ignore zero/zero or one/zero
    
    histogram_list = []
   
    x =get_histogram_bins_time(whole_RPKM,geneA,geneB,sample_index,sample_size)
    histogram_list.append(x)
    #add_self_image
    x = get_histogram_bins_time(whole_RPKM,geneA,geneA,sample_index,sample_size)
    histogram_list.append(x)
    x = get_histogram_bins_time(whole_RPKM,geneB,geneB,sample_index,sample_size)
    histogram_list.append(x)

        
    index = int(np.where(gene_map==geneA)[0])
    cov_list_geneA = cov_matrix[index, :]
    cov_list_geneA = cov_list_geneA.ravel()   
    
    cov_list_geneA = np.delete(cov_list_geneA,train_known_index) # avoid leak
    sub_gene_map = np.delete(gene_map,train_known_index)
    
    the_order = np.argsort(-cov_list_geneA)
    select_index = the_order[0:10] # top_cov number
  
    
    for j in select_index:           
        x = get_histogram_bins_time(whole_RPKM,geneA,sub_gene_map[j],sample_index,sample_size)
        histogram_list.append(x)
    
    
    indexB = int(np.where(gene_map==geneB)[0])
    cov_list_geneB = cov_matrix[indexB, :]
    cov_list_geneB = cov_list_geneB.ravel()

    cov_list_geneB = np.delete(cov_list_geneB,train_known_index) # avoid leak    
   
    the_order = np.argsort(-cov_list_geneB)
    select_index = the_order[0:10] # top_cov number
    
    for j in select_index:       
        x = get_histogram_bins_time(whole_RPKM,sub_gene_map[j],geneB,sample_index,sample_size)
        histogram_list.append(x)
        
    return histogram_list
 
def debug_print_neighbor_genename(geneA,geneB,cov_matrix,gene_map,whole_RPKM,sample_index,sample_size, train_known_index):
    np.seterr(divide='ignore',invalid='ignore') # ignore zero/zero or one/zero
          
    index = int(np.where(gene_map==geneA)[0])
    cov_list_geneA = cov_matrix[index, :]
    cov_list_geneA = cov_list_geneA.ravel()   
    
    cov_list_geneA = np.delete(cov_list_geneA,train_known_index) # avoid leak
    sub_gene_map = np.delete(gene_map,train_known_index)
    
    the_order = np.argsort(-cov_list_geneA)
    select_index = the_order[0:10] # top_cov number
    GeneA_neighbor_list=sub_gene_map[select_index]

    indexB = int(np.where(gene_map==geneB)[0])
    cov_list_geneB = cov_matrix[indexB, :]
    cov_list_geneB = cov_list_geneB.ravel()
    cov_list_geneB = np.delete(cov_list_geneB,train_known_index) # avoid leak    
    the_order = np.argsort(-cov_list_geneB)
    select_index = the_order[0:10] # top_cov number
    GeneB_neighbor_list=sub_gene_map[select_index]
      
    return GeneA_neighbor_list, GeneB_neighbor_list

def debug_print_neighbor_genename_one(geneA,cov_matrix,gene_map,whole_RPKM,train_known_index):
    np.seterr(divide='ignore',invalid='ignore') # ignore zero/zero or one/zero
          
    index = int(np.where(gene_map==geneA)[0])
    cov_list_geneA = cov_matrix[index, :]
    cov_list_geneA = cov_list_geneA.ravel()   
    
    cov_list_geneA = np.delete(cov_list_geneA,train_known_index) # avoid leak
    sub_gene_map = np.delete(gene_map,train_known_index)
    
    the_order = np.argsort(-cov_list_geneA)
    select_index = the_order[0:10] # top_cov number
    GeneA_neighbor_list=sub_gene_map[select_index]
     
    return GeneA_neighbor_list        
        
        
if __name__ == '__main__':


    datadir= args.genepairs_filepath
    expr_dir = args.expression_filepath
    n_tp = args.n_timepoints
    save_header = args.save_dir
    save_name = args.save_filename
    shared_known_genes = args.func_geneset_filepath
    G1,G2,Label = load_gene_pair(datadir)
   
    sample_index, sample_size, whole_RPKM , gene_map= load_real_data(expr_dir,3) 

    known_gene = np.load(shared_known_genes)
    
    known_gene_index = []
    for i in range(len(known_gene_index)):
        known_gene_index.append(int(np.where(gene_map==known_gene[i])[0]))
    
    cov_matrix = calculate_cov(whole_RPKM)
    x_data = []
    y_data = []
    for i in range(len(Label)):
    #for i in range(2): # for debug
        print('%d/%d'%(i+1,len(Label)))
        hist_list = get_images_with_top_cov_pairs(G1[i],G2[i],cov_matrix,gene_map,whole_RPKM,sample_index,sample_size,known_gene_index)
        x_data.append(hist_list)
        y_data.append(Label[i])
    
    xx = np.array(x_data)

    if save_header is not None:
        if not os.path.isdir(save_header):
            os.mkdir(save_header)
    np.save(save_header + '/' + save_name,xx)

