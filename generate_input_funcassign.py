import pandas as pd
import numpy as np
#import pickle
import argparse
import os

parser = argparse.ArgumentParser(description="")
parser.add_argument('-genepairs_filepath',required=True, default=None, help="gene pairs file. format: gene1 gene2 label")
parser.add_argument('-expression_filepath',required=True, default=None, help=".h5 file. scRNA-seq gene expression")
parser.add_argument('-func_geneset_filepath',required=True, default=None, help="The shared gene between download genes and the genes in expression profiles")
parser.add_argument('-n_timepoints',required=True, default=None, help="The number of time points")
parser.add_argument('-save_dir',required=True, default=None, help="The path of output files")
parser.add_argument('-save_filename',type=str,required=True, default=None, help="The name of output file")
parser.add_argument('-random_geneset_filepath', default=None, help="The remaining genes in expression dataset but exclused the functional genes")
parser.add_argument('-neighbor_criteria', type=str, default='cov', help="The option of the measures used to find out the neighbor genes ('cov': covariance; 'corr':correlation, default 'cov').")
parser.add_argument('-image_resolution', type=int, default=8, help="Image resolution (default 8).")
parser.add_argument('-top_num', type=int, default=10, help="The number of top neighbor genes to be involved (default 10).")
parser.add_argument('-get_abs', type=bool, default=False, help="Apply absolute value (covariance/correlation) to identify neighbor genes? (default False).")


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
            store = pd.HDFStore(filedir+'/'+'ST_t'+str(indexy)+'.h5') #    # scRNA-seq expression data                        )#
            rpkm = store['STrans']            
            store.close()
            total_RPKM_list.append(rpkm)
            sample_size_list = sample_size_list + [indexy for i in range (rpkm.shape[0])] #append(rpkm.shape[0])
            sample_sizex.append(rpkm.shape[0])
            samples = np.array(sample_size_list)
            
        total_RPKM = pd.concat(total_RPKM_list, ignore_index=True)
        total_RPKM.columns = total_RPKM.columns.map(lambda x:x.lower())
        gene_map = total_RPKM.columns.values
        return samples,sample_sizex,total_RPKM, gene_map
        
        
        
def get_histogram_bins_time(RPKM, geneA, geneB, samples, sample_size):
    num_timepoints = len(sample_size)
    x_geneA=RPKM[geneA]
    x_geneB=RPKM[geneB]
    
    if len(x_geneA.shape) > 1:
        x_geneA = x_geneA.iloc[:,0]
 
    if len(x_geneB.shape) > 1:
        x_geneB = x_geneB.iloc[:,0]
 
    H =[]
    if x_geneA is not None:
        if x_geneB is not None:
            
            #x_1 = np.array(np.log10(x_geneA + 10 ** -2)) 
            #x_2 = np.array(np.log10(x_geneB + 10 ** -2))
            x_1 = x_geneA 
            x_2 = x_geneB   
         
            datax = np.concatenate((x_1[:, np.newaxis], x_2[:, np.newaxis], samples[:, np.newaxis]), axis=1)           
            H, edges = np.histogramdd(datax, bins=(args.image_resolution, args.image_resolution, num_timepoints))                    
            HT = (np.log10(H / sample_size + 10 ** -3) + 3) / 3
            H2 = np.transpose(HT, (2, 0, 1))            
            return H2
        else:
            return None
    else:
        return None
  
  
  
def calculate_cov(rpkm, measure='cov'):
    expr = rpkm.iloc[:][:]   
    expr = np.asarray(expr)  
    expr = expr.transpose()   
    ## for sparse data
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components = 100)
    # pca.fit(expr)
    # expr = pca.transform(expr)
    if measure=='cov':
        matrix = np.cov(expr)
    elif measure=='corr':
        matrix = np.corrcoef(expr)
    else:
        measure=  None
        print('Check the input of neighbor_criteria.')
    return matrix

    
def get_images_with_top_cov_pairs(geneA,geneB,cov_matrix,gene_map,whole_RPKM,sample_index,sample_size, train_known_index, random_gene_index):
    np.seterr(divide='ignore',invalid='ignore') # ignore zero/zero or one/zero
    
    histogram_list = []
    # remove the functional genes when selecting neigbor genes
    sub_gene_map = np.delete(gene_map,train_known_index)
    # primary image
    x = get_histogram_bins_time(whole_RPKM,geneA,geneB,sample_index,sample_size) 
    histogram_list.append(x)
    # self_image
    x = get_histogram_bins_time(whole_RPKM,geneA,geneA,sample_index,sample_size)
    histogram_list.append(x)
    x = get_histogram_bins_time(whole_RPKM,geneB,geneB,sample_index,sample_size)
    histogram_list.append(x)



    index = np.where(gene_map==geneA)[0]
    if len(index)>1:
        index = int(index[0])
    else:
        index = int(index)
    cov_list_geneA = cov_matrix[index, :]
    cov_list_geneA = cov_list_geneA.ravel()    
    cov_list_geneA = np.delete(cov_list_geneA, train_known_index) # When identifying neighbor genes, we exclude the genes with the annotataion to avoid leak training information       
    if args.get_abs:
        cov_list_geneA = np.abs(cov_list_geneA)
    the_order = np.argsort(-cov_list_geneA) # descending 
    select_index = the_order[0: args.top_num] # the indexes of top-n neighbor genes for geneA
    # neighbor images of geneA
    for j in select_index:           
        x = get_histogram_bins_time(whole_RPKM,geneA,sub_gene_map[j],sample_index,sample_size)
        histogram_list.append(x)
    
    
    #indexB = int(np.where(gene_map==geneB)[0])
    indexB = np.where(gene_map==geneB)[0]
    if len(indexB)>1:
        indexB = int(indexB[0])
    else:
        indexB = int(indexB)
    cov_list_geneB = cov_matrix[indexB, :]
    cov_list_geneB = cov_list_geneB.ravel()
    cov_list_geneB = np.delete(cov_list_geneB,train_known_index) # avoid leak training information  
    if args.get_abs:
        cov_list_geneB = np.abs(cov_list_geneB)
    the_order = np.argsort(-cov_list_geneB)
    select_index = the_order[0: args.top_num] 
    # neighbor images of geneB
    for j in select_index:       
        x = get_histogram_bins_time(whole_RPKM,sub_gene_map[j],geneB,sample_index,sample_size)
        histogram_list.append(x)
          
    return histogram_list
 
        
        
if __name__ == '__main__':
    
    # user input
    datadir= args.genepairs_filepath
    expr_dir = args.expression_filepath
    n_tp = args.n_timepoints
    save_header = args.save_dir
    save_name = args.save_filename
    shared_known_genes = args.func_geneset_filepath
    random_genes = args.random_geneset_filepath
    neighbor_criteria = args.neighbor_criteria
    # Load gene pairs and its labels
    G1,G2,Label = load_gene_pair(datadir)
    # Load scRNA-seq gene expression data (have been normalized by sctransform)
    sample_index, sample_size, whole_RPKM , gene_map= load_real_data(expr_dir,3) 
    # Load known genes (the genes with the annotation), the neighbor genes will exclude these genes
    known_gene = np.load(shared_known_genes)
    # Load randomly selected genes (the genes without the annotation, expressed in the scRNA-seq dataset)
    random_gene = np.load(random_genes)
    
    
    # The indexes of genes in the expression dataset
    known_gene_index = []
    for i in range(len(known_gene)):
        tmp = np.where(gene_map==known_gene[i])[0]
        if len(tmp) > 1:
            tmp = tmp[0]
        known_gene_index.append(int(tmp))
    
    random_gene_index = []
    for i in range(len(known_gene)):
        tmp = np.where(gene_map==random_gene[i])[0]
        if len(tmp) > 1:
            tmp = tmp[0]
        random_gene_index.append(int(tmp))
   
    # get covarian/correlation matrix of the genes
    cov_matrix = calculate_cov(whole_RPKM, measure=neighbor_criteria)
 
    ## for dynDeepDRIM
    x_data = []
    y_data = []
    for i in range(len(Label)):
    #for i in range(2): # for debug      
        print('%d/%d'%(i+1,len(Label)))
        hist_list = get_images_with_top_cov_pairs(G1[i],G2[i],cov_matrix,gene_map,whole_RPKM,sample_index,sample_size,known_gene_index,random_gene_index)
        x_data.append(hist_list)
        y_data.append(Label[i])
    
    xx = np.array(x_data)

    if save_header is not None:
        if not os.path.isdir(save_header):
            os.mkdir(save_header)
    np.save(os.path.join(save_header, save_name), xx)


    ## for TDL models, comparison experiment only 
    # x_data = []
    # y_data = []
    # for i in range(len(Label)):
        # print('%d/%d'%(i+1,len(Label)))
        # Histogram_3D = get_histogram_bins_time(whole_RPKM,G1[i],G2[i],sample_index,sample_size)
        # x_data.append(Histogram_3D)
        # y_data.append(Label[i])
        
    # xx = np.array(x_data)[:,:,:,:, np.newaxis]

    # if save_header is not None:
        # if not os.path.isdir(save_header):
            # os.mkdir(save_header)
    # np.save(save_header + '/' + save_name + '_vTDL',xx)

    