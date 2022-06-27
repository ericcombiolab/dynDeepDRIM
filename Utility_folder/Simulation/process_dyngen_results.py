import pandas as pd
import numpy as np
import os
import re


def get_cell_timepoint_info(data_dir):
    df_cell_info = pd.read_table(os.path.join(data_dir,'cell_info.txt'),sep='\s+')
    df_cell_timepoint = df_cell_info[['cell_id', 'timepoint_group']]
    return df_cell_timepoint
  
  
def get_GRN_info(data_dir):
    df_grn_info = pd.read_table(os.path.join(data_dir,'grn_info.txt'),sep='\s+')
    df_regulate = df_grn_info[['from','to']]
    return df_regulate
   
   
def identify_TF_gene_pairs(df_regulate):
    geneA_list = df_regulate['from'].values

    flag_TF = []
    for geneA in geneA_list:
        if re.match(r'Target*',geneA) or re.match(r'HK*',geneA):
            flag_TF.append(0)
        else:
            flag_TF.append(1)
    flag_TF = np.array(flag_TF)   
    TF_gene_index = np.argwhere(flag_TF == 1).ravel()
    return TF_gene_index
 
 
def identify_HK_genes(df_regulate): 
    geneA_list = df_regulate['from'].values
    geneB_list = df_regulate['to'].values

    hk_genes = []
    for geneA in geneA_list:
        if re.match(r'HK*',geneA):
            hk_genes.append(geneA)
  
    for geneB in geneB_list:
        if re.match(r'HK*',geneB):
            hk_genes.append(geneB)

    hk_genes = list(set(hk_genes))
    return hk_genes  
  
  
def get_sorted_expr_with_timeInterval(data_dir, counts, tp_info):
    counts_with_time = counts.T
    counts_with_time['time'] = tp_info['timepoint_group'].values
    counts_sorted_by_time = counts_with_time.sort_values(by=['time']) # the sorted counts
    
    tmp_time = counts_sorted_by_time['time'].values.ravel()
    n_cells_each_tp = [] 
    for i in range(n_timepoints):
        index_one_time = np.argwhere(tmp_time == i).ravel()
        n_cells_each_tp.append(len(index_one_time))
    
    return counts_sorted_by_time, n_cells_each_tp 
 
 
def save_txt_single_col(data, path):
    if type(data) != list:
        print('data should be in list format!')
        return None
        
    f =open(path,'w')
    for i in data:
        f.write(str(i)+'\n')
    f.close()


def pairs_join_2_oneString(gene_pairs, low_case = False):
    collect = []
    for i in range(len(gene_pairs)):
        if low_case == True:
            string = '\t'.join(gene_pairs[i])
            collect.append(string.lower())
        else:
            collect.append('\t'.join(gene_pairs[i]))
    return collect           


def generate_transitive_pairs(df_regulate, TF_target):
    

    _geneB = list(set(TF_target[:,1]))
    _geneB.sort(key=TF_target[:,1].tolist().index)
    
    ## identify TF->TF pairs, remove TF->target
    geneB = []
    for g in _geneB:
        if not re.match(r'Target*',g):           
            geneB.append(g)
 

    gene_pairs_all = df_regulate.values
    geneAs_all = df_regulate['from'].values
    geneBs_all = df_regulate['to'].values

    transitive_pairs = []
    for i in range(len(geneB)):      
        if geneB[i] in geneAs_all:
            idx =np.argwhere(geneAs_all == geneB[i]).ravel()
            _geneB_2_C = gene_pairs_all[idx]     
            geneB_2_C = [] ## identify TF->Target pairs, remove TF->TF
            for j in range(len(_geneB_2_C)):
                if re.match(r'Target*', _geneB_2_C[j,1]):           
                    geneB_2_C.append(_geneB_2_C[j].tolist())
            geneB_2_C = np.array(geneB_2_C)
 
     
            idx =np.argwhere(geneBs_all == geneB[i]).ravel()      
            geneA_2_B = gene_pairs_all[idx]
           
            # idx =np.argwhere(TF_target == geneB[i]).ravel()
            # geneA_2_B = TF_target[idx]
            
            
            # A->B->C (transitive interaction)          
            for j in range(len(geneA_2_B)):
                gA= geneA_2_B[j,0]
                for k in range(len(geneB_2_C)):
                    gC = geneB_2_C[k,1]
                    transitive_pairs.append([gA,gC])
      
    # remove TF-target pairs in A->C pool      
    TF_target_pairs = pairs_join_2_oneString(TF_target)
    Transitive_pairs = pairs_join_2_oneString(transitive_pairs) 
    union = set(TF_target_pairs) & set(Transitive_pairs)
    final_transitive_pairs = list(set(Transitive_pairs) - union)
   
    return final_transitive_pairs


def get_non_interaction_gene_pool_withoutHKgenes(transitive, tf_target, df_regulate):
    collect = []
    for pairs in transitive:
       gA, gB = pairs.split('\t')
       collect.append(gB)
    interaction_pool = set(tf_target[:,1].tolist() + collect)
    non_interaction_pool = (set(df_regulate['from']) | set(df_regulate['to'])) - interaction_pool
    return list(non_interaction_pool)


def generate_benchmark_pairs(transitive_regu_pairs, regulate_gene_pairs, non_inx_genepool):
    collect = []    ## tansitive info 
    collect_ = []
    for pairs in transitive_regu_pairs:
       gA, gB = pairs.split('\t')
       collect.append(gA)
       collect_.append([gA, gB])
    collect = np.array(collect)
    collect_ = np.array(collect_)
    
    TF_target = regulate_gene_pairs.tolist() ## tf-target info 
    TFs = regulate_gene_pairs[:,0].tolist()
    TF_target.sort()
    TFs.sort()
    TFs_ = list(set(TFs))
    TFs_.sort(key= TFs.index)
    
    TFs = np.array(TFs)
    TF_target = np.array(TF_target)
    pair_scopes = []
    benchmark_pairs = []    
    aa =0
    
    for TF in TFs_:
    
        if TF in collect:  ## negative sample
            index =np.argwhere(collect == TF).ravel() ## transitive intx pool
            TF_transitive_pairs = collect_[index]           
        else:
            TF_transitive_pairs = []
        
        non_interx_pairs = [] ## non-interactive pool
        for gene in non_inx_genepool:
            non_interx_pairs.append([TF, gene])
        non_interx_pairs = np.array(non_interx_pairs)

        
        idx= np.argwhere(TFs == TF).ravel() ## positive sample
        pair_scopes.append(len(idx) * 2)
        
        # print(len(TF_target[idx]))
        # print(len(TF_transitive_pairs))
        # print(len(non_interx_pairs))
        if len(non_interx_pairs) + len(TF_transitive_pairs) < len(TF_target[idx]):
            print('negative samples pool smaller than positive samples')

            import sys
            sys.exit()
            
            
        ## benchmark pairs generation
        for i in range(len(TF_target[idx])):         
        
            benchmark_pairs.append(TF_target[idx][i].tolist())
            
            if (i%2 == 0) and (len(TF_transitive_pairs)>0):
                tmp_idx = np.random.randint(0,len(TF_transitive_pairs))                
                benchmark_pairs.append(TF_transitive_pairs[tmp_idx].tolist())
                TF_transitive_pairs = np.delete(TF_transitive_pairs, tmp_idx, axis=0)
            else:
        
                tmp_idx = np.random.randint(0,len(non_interx_pairs))
                              
                benchmark_pairs.append(non_interx_pairs[tmp_idx].tolist())
                non_interx_pairs = np.delete(non_interx_pairs, tmp_idx, axis=0)
                
        # ## debug        
        # if aa>1:
            # break
        # aa+=1
        
    return benchmark_pairs, pair_scopes

if __name__ == '__main__':
   
    n_timepoints = 4
    simulation_data_dir = './100TFs_20000cells_5000hk_4tp'
     
     
    ############################## Getting time-sorted counts matrix ####################################
    if True:
        ### load raw counts
        counts = pd.read_table(os.path.join(simulation_data_dir,'counts.txt'),sep='\s+',low_memory=False) 
        gene_list = np.array(counts.index)
        
        f = open(os.path.join(simulation_data_dir,'simulation_gene_list.txt'), 'w')
        for gene in gene_list:
            f.write(gene.lower() + '\t' + gene.lower() + '\n')
        f.close()
    

        ### cells sorted by time
        tp_info = get_cell_timepoint_info(simulation_data_dir)
        sorted_counts, time_interval = get_sorted_expr_with_timeInterval(simulation_data_dir, counts, tp_info)
        sorted_counts = sorted_counts.iloc[:,:-1].T # remove timepoint column

        sorted_counts.to_csv(os.path.join(simulation_data_dir, 'sorted_counts.txt'),sep='\t')
        save_txt_single_col(time_interval,os.path.join(simulation_data_dir, 'time_interval.txt'))




    ############################## Process GRN data ####################################    
    if True:     
        df_regulate = get_GRN_info(simulation_data_dir)   # direct regulation (TF->target, target->target)  
        
        ## extract TF-target pairs
        index_tf_gene = identify_TF_gene_pairs(df_regulate) # direct regulation (TF->gene)
        regulate_gene_pairs = df_regulate.iloc[index_tf_gene].values
        for_saving = pairs_join_2_oneString(regulate_gene_pairs)
        save_txt_single_col(for_saving, os.path.join(simulation_data_dir,'TFtarget_pairs.txt'))
        

        ## transitive pairs (degree=1, TF1->TF2->g2, TF1->g2 was identified as transitive interaction)
        transitive_regu_pairs = generate_transitive_pairs(df_regulate, regulate_gene_pairs) 
        save_txt_single_col(transitive_regu_pairs, os.path.join(simulation_data_dir,'transitive_pairs.txt'))    

        ## non-interacted genes (not be directly regulated by TF) ,,,,, this need to be improve(hk genes?/for specific TF)
        #non_inx_genepool = get_non_interaction_gene_pool_withoutHKgenes(transitive_regu_pairs, regulate_gene_pairs, df_regulate)
        non_inx_genepool = identify_HK_genes(df_regulate)
        save_txt_single_col(non_inx_genepool, os.path.join(simulation_data_dir,'non_ix_genepool.txt'))

        
        ## generate benchmark pairs with label
        benchmark, scopes = generate_benchmark_pairs(transitive_regu_pairs, regulate_gene_pairs, non_inx_genepool)
        for_saving = pairs_join_2_oneString(benchmark, low_case=True)
        benchmark_label = []
        for i in range(int(len(for_saving)/2)):  
            benchmark_label.append(for_saving[2*i] + '\t1')
            benchmark_label.append(for_saving[2*i+1] + '\t0')
          
        save_txt_single_col(benchmark_label, os.path.join(simulation_data_dir,'simulation_gene_pairs.txt'))
        
        pos = [0]  
        for i in range(len(scopes)):
            offset = scopes[i]
            pos.append( pos[-1] + offset )
        save_txt_single_col(pos, os.path.join(simulation_data_dir,'simulation_gene_pairs_num.txt'))
        
        
        
       
    

