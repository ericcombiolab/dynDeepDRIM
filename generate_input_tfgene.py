
from __future__ import print_function


import pandas as pd
from numpy import *
import numpy as np
import json, re,os, sys
import argparse

parser = argparse.ArgumentParser(description="")

parser.add_argument('-out_dir', required=True, help='Indicate the path for output.')
parser.add_argument('-expr_file', required=True, help='The file of the gene expression profile. Can be h5 or csv file, the format please refer the example data.')
parser.add_argument('-pairs_for_predict_file', required=True, help='The file of the training gene pairs and their labels.')
parser.add_argument('-geneName_map_file', required=True, default=None, help='The file to map the name of gene in expr_file to the pairs_for_predict_file')
parser.add_argument('-flag_load_split_batch_pos', default=True, help='Is there a file that indicate the position in pairs_for_predict_file to divide pairs into different TFs.')
parser.add_argument('-TF_divide_pos_file', default=None, help='File that indicate the position in pairs_for_predict_file to divide pairs into different TFs.')
parser.add_argument('-TF_num', type=int, default=None, help='To generate representation for this number of TFs. Should be a integer that equal or samller than the number of TFs in the pairs_for_predict_file.')
parser.add_argument('-n_timepoints', type=int,default=1, help='The number of time points we plan to use for time course data')

args = parser.parse_args()


class RepresentationTest2:
    def __init__(self,output_dir,x_method_version=1, max_col=None, start_batch_num=0,
                 end_batch_num=None,load_batch_split_pos=False):
        # input
        self.load_batch_split_pos = load_batch_split_pos
        self.geneIDs = None  #
        self.rpkm = None
        self.geneID_map = None  # not necessary, ID in expr to ID in gold standard
        self.ID_to_name_map = None
        self.split_batch_pos = None
        self.gold_standard = {}  # geneA;geneB -> 0,1,2 #note direction, geneA,geneB is diff with geneB,geneA
        self.output_dir = output_dir 
        self.x_method_version = x_method_version

        self.max_col = max_col
        self.key_list = []
        self.generate_key_list = []
        #
        self.sample_size = 0
        self.sample = None
        self.total_RPKM = None
        self.timepoints = 1
        self.sample_sizex = []
  
        self.resolution_3d = 8

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
            os.mkdir(output_dir+"v_dynDeepDRIM")

        self.top_num = None
        self.add_self_image = None
        self.get_abs = None
        self.end_batch_num= end_batch_num
        self.start_batch_num = start_batch_num
        self.cov_matrix = None
        self.corr_matrix = None


    def get_expr_by_networki_geneName(self, geneA):  #for simulation, for liver
        index=self.geneID_map.get(str(geneA)) # gene symbol

        if index is None:
            index = int(geneA)
            geneA_x = self.rpkm.iloc[:,index]

        else:
            #index=int(index)
            if type(self.geneIDs[0])==int:
                index=int(index)
            index2 = np.where(self.geneIDs==index) 
            index = index2[0]
            geneA_x = self.rpkm.iloc[:,index] 

        geneA_x=geneA_x.to_numpy().reshape(-1)
        return geneA_x

    def get_index_by_networki_geneName(self, geneA):  #for simulation, for liver
        index=self.geneID_map.get(str(geneA))
        if index is None:
            index = int(geneA)
            #print("gene", geneA, "not found")
            #return None
        else:
            if type(self.geneIDs[0])==int:
                index=int(index)
            index2 = np.where(self.geneIDs==index)
            index = index2[0]
        return index

    def get_gene_list(self, file_name):
        import re
        h = {}
        h2 = {}
        s = open(file_name, 'r')  # gene symbol ID list of sc RNA-seq
        for line in s:
            search_result = re.search(r'^([^\s]+)\s+([^\s]+)', line)
            #print("-",search_result.group(1),"-",search_result.group(2),"-")
            h[str(search_result.group(1).lower())] = str(search_result.group(2).lower())  # h [gene symbol] = gene ID
            h2[str(search_result.group(2).lower())] = str(search_result.group(1).lower()) #h2 geneID = gene symbol
        self.geneID_map = h
        self.ID_to_name_map = h2
        #print(self.geneID_map)
        s.close()

    def load_real_data(self, filedir, n_timepoints):
        
        if not filedir == 'None' and int(n_timepoints):
            sample_size_list = []
            total_RPKM_list = []
            for indexy in range (0,int(n_timepoints)): # time points number
                store = pd.HDFStore(filedir+'/'+'RPKM_'+str(indexy)+'.h5') #    # scRNA-seq expression data                        )#
                rpkm = store['RPKMs']
                #for mesc1 dataset
                # store = pd.HDFStore(filedir+'/'+'RPM_'+str(indexy)+'.h5')
                # rpkm = store['/RPKM']
                store.close()
                total_RPKM_list.append(rpkm)
                sample_size_list = sample_size_list + [indexy for i in range (rpkm.shape[0])] #append(rpkm.shape[0])
                self.sample_sizex.append(rpkm.shape[0])
                self.samples = np.array(sample_size_list)
                self.sample_size = len(sample_size_list)
            self.total_RPKM = pd.concat(total_RPKM_list, ignore_index=True)
            print('\n')
            print('----------- real expression data info ------------')
            print('read sc RNA-seq expression, size(n_cell):',self.sample_size) # YU: sample_size= sum(n_rows_all)
            print('sample size in eachtime(n_cell_time):',self.sample_sizex)
            print('table for all time-series expression data',self.total_RPKM)
            print('sample time point symbol:', self.samples) # YU : a list with same length of n_rows_all, indicates cell data from which time point.
            

        self.rpkm = self.total_RPKM # for compatible other funcs
        self.geneIDs=self.total_RPKM.columns
        self.geneIDs=np.asarray(self.geneIDs,dtype=str)

        print("gene nums:",len(self.total_RPKM.columns))
        print("cell nums:", self.sample_size)
        print('\n')

    def get_gold_standard(self,filename):
        unique_keys={}
        s = open(filename)   ### read 'the gene pair and label' file
        for line in s:
            separation = line.split()
            geneA_name, geneB_name, label = separation[0], separation[1], separation[2]
            geneA_name = geneA_name.lower()
            geneB_name = geneB_name.lower()
            key=str(geneA_name)+","+str(geneB_name)
            key2=str(geneB_name)+","+str(geneA_name)
            #if key not in self.gold_standard:
                #if key2 not in self.gold_standard:
            if self.load_batch_split_pos:
                if key in self.gold_standard.keys():
                    if label == int(2):
                        pass
                    else:
                        self.gold_standard[key] = int(label)
                    self.key_list.append(key)
                else:
                    self.gold_standard[key] = int(label)
                    self.key_list.append(key)
            else:
                if int(label) != 2:
                    unique_keys[geneA_name] = self.geneID_map.get(geneA_name)
                    unique_keys[geneB_name] = self.geneID_map.get(geneB_name)

                    if geneA_name in unique_keys:
                        if geneB_name in unique_keys:
                            print(key,label,int(label))
                            self.gold_standard[key] = int(label)
        s.close()
       
    def load_split_batch_pos(self,filename):
        self.split_batch_pos = []
        s = open(filename) 
        for line in s:
            separation = line.split()
            #print("line",line)
            #print("separation",separation)
            self.split_batch_pos.append(separation[0])  

        print('load pairs batch(TF_index) : ',np.array(self.split_batch_pos))   
        print('\n')
        s.close()

    # new for time-series data , this is the exactly defference of constructing histogram(image) 
    def get_histogram_bins_time(self, geneA, geneB):
        x_geneA=self.get_expr_by_networki_geneName(geneA)
        x_geneB=self.get_expr_by_networki_geneName(geneB)
        H =[]
        if x_geneA is not None:
            if x_geneB is not None:
                #print("x_geneA",x_geneA)
                #print("x_geneB",x_geneB)
                x_tf = np.array(np.log10(x_geneA + 10 ** -2) )  # to log space
                x_gene = np.array(np.log10(x_geneB + 10 ** -2))
             
                datax = np.concatenate((x_tf[:, np.newaxis], x_gene[:, np.newaxis], self.samples[:, np.newaxis]), axis=1)           
                H, edges = np.histogramdd(datax, bins=(self.resolution_3d, self.resolution_3d, self.timepoints))
                           
                HT = (np.log10(H / self.sample_sizex + 10 ** -3) + 3) / 3
                H2 = transpose(HT, (2, 0, 1))
                
                return H2
            else:
                return None
        else:
            return None

    def get_gene_pair_data_time(self,geneA,geneB, x_method_version):
        # input geneA, geneB, get corresponding expr and get histogram
        # return x, y, z
        if self.x_method_version != 0:
            if self.max_col is None:
                self.max_col = 2 * len(self.geneIDs)


        x = self.get_x_for_one_pair_version11_time(geneA, geneB)           

        if x is not None:
            key = str(geneA)+','+str(geneB)
            y = self.gold_standard.get(key)      
            z = key    
            #print("x.shape", x.shape)
         
            return [x,y,z]
        else:
            print('Error: 3D histogram generated failed')
            sys.exit()
            

    def get_batch_time(self,gene_list,save_header,x_method_version):
        xdata = []  # numpy arrary [k,:,:,1], k is number o fpairs
        ydata = []  # numpy arrary shape k,1
        zdata = []  # record corresponding pairs

        if len(gene_list)>0: # batch gene pair list 
            for i in range(0,len(gene_list)):  # YU: process a batch gene pair data
            
                geneA=gene_list[i].split(',')[0]
                geneB=gene_list[i].split(',')[1]
                key = str(geneA) + ',' + str(geneB)
                gold_y = self.gold_standard.get(key) # YU: orignal label from ChIP
             
                [x,y,z] = self.get_gene_pair_data_time(geneA,geneB,x_method_version)   
                if x is not None:
                    xdata.append(x)
                    ydata.append(y)
                    zdata.append(z)
                           
               #debug only
                if i>=10:
                    break
                   
            if (len(xdata) > 0):
                if len(shape(xdata)) == 5:
                    xx = xdata
                elif len(shape(xdata)) == 4:
                    xx = np.array(xdata)[:, :, :, :, np.newaxis] # new channel(E.g. RGB 3 channels)
                else:
                    xx = np.array(xdata)[:, :, :, :, np.newaxis]

            print("Time series histogram data shape",shape(xx))
            print("save in:",save_header)
            print('\n')
            np.save(save_header+'_xdata.npy',xx)
            np.save(save_header + '_ydata.npy', np.array(ydata))
            np.save(save_header + '_zdata.npy', np.array(zdata))


    def get_train_test(self,batch_index=None, generate_multi=True,
                       TF_pairs_num_lower_bound=0,TF_pairs_num_upper_bound=None):
        #deal with cross validation or train test batch partition,mini_batch
        print("------------Data process info-------------")
        self.generate_key_list=[]
        if self.split_batch_pos is not None:
            #from collections import OrderedDict
            #self.gold_standard = OrderedDict(self.gold_standard)
            key_list = self.key_list
        else:
            from collections import OrderedDict
            self.gold_standard = OrderedDict(self.gold_standard)
            key_list = list(self.gold_standard.keys())
            #key_list = list(sorted(self.gold_standard.keys()))

        print("gold standard len(equal with n_pairs if using all gene pairs):",len(key_list))
        if self.split_batch_pos is not None:

            print('tf intervals(TF_num/n_batch):', len(self.split_batch_pos)-1)
            print('minimum TF paris number in a batch:',TF_pairs_num_lower_bound)
            print('maximum TF paris number in a batch:',TF_pairs_num_upper_bound)
            print('\n')

            index_start_list=[]
            index_end_list=[]

            for i in range(0, (len(self.split_batch_pos)-1)):
                index_start = int(self.split_batch_pos[i])  # YU: split the genepair set to be batches (interval start index, end index)
                index_end = int(self.split_batch_pos[i + 1])

                if (index_end-index_start)>=TF_pairs_num_lower_bound:
                    if TF_pairs_num_upper_bound is None:
                        index_start_list.append(index_start)
                        index_end_list.append(index_end)
                    else:
                        if (index_end-index_start)<=TF_pairs_num_upper_bound:
                            index_start_list.append(index_start)
                            index_end_list.append(index_end)
                            

            if self.end_batch_num is None:
                self.end_batch_num = len(index_start_list)
            else:
                if self.end_batch_num > len(index_start_list):
                    self.end_batch_num = len(index_start_list) ####?????!!!!!

            for i in range(self.start_batch_num, self.end_batch_num):
                print('Batch-th:',i+1)
                index_start = int(index_start_list[i])
                index_end = int(index_end_list[i])
                # print("index_start",index_start)
                # print("index_end", index_end)
                if index_end <= len(key_list):
                    select_list = list(key_list[j] for j in range(index_start, index_end)) # batch gene pair
                    
                    for j in range(index_start,index_end):
                        self.generate_key_list.append(key_list[j]+','+str(self.gold_standard.get(key_list[j])))

                    self.resolution_3d=8
                    self.get_batch_time(select_list, self.output_dir + "v_dynDeepDRIM/" + str(i), 11)# for D_T

                #debug only
                if i>=2:
                    sys.exit()
        else:
            sys.exit('Pls provide split_batch_pos file!')


    def get_top_cov_pairs(self,geneA,geneB,cov_or_corr="cov"):
        np.seterr(divide='ignore',invalid='ignore') # ignore zero/zero or one/zero
        # get cov value first
    
        if self.corr_matrix is None or self.cov_matrix is None:
            self.calculate_cov()
        if cov_or_corr=="corr":
            np.fill_diagonal(self.corr_matrix, 0)
              
        histogram_list = []
        networki = geneA.split(":")[0]
       
        x = self.get_histogram_bins_time(geneA, geneA)
           
        if self.add_self_image:
            histogram_list.append(x)
    
        x = self.get_histogram_bins_time(geneB, geneB)
      
        if self.add_self_image:
            histogram_list.append(x)
            
        index = self.get_index_by_networki_geneName(geneA)

        if cov_or_corr=="cov":
            cov_list_geneA = self.cov_matrix[index, :]
        else:
            cov_list_geneA = self.corr_matrix[index, :]
        cov_list_geneA = cov_list_geneA.ravel() # flatten     
        if self.get_abs:
            cov_list_geneA = np.abs(cov_list_geneA)
        the_order = np.argsort(-cov_list_geneA)
        select_index = the_order[0:self.top_num]
        
        for j in select_index:    
            x = self.get_histogram_bins_time(geneA, str(j))         
            histogram_list.append(x)
        
        ####
        indexB = self.get_index_by_networki_geneName(geneB)
        if cov_or_corr=="cov":
            cov_list_geneB = self.cov_matrix[indexB, :]
        else:
            cov_list_geneB = self.corr_matrix[indexB, :]
        cov_list_geneB = cov_list_geneB.ravel()
        if self.get_abs:
            cov_list_geneB = np.abs(cov_list_geneB)
        the_order = np.argsort(-cov_list_geneB)
        select_index = the_order[0:self.top_num]
        for j in select_index: 
            x = self.get_histogram_bins_time(str(j), geneB)
            histogram_list.append(x)
        #print('multi-3D-histogram shape:',np.array(histogram_list).shape)
     
        return histogram_list


    def calculate_cov(self):
        expr = self.rpkm.iloc[:][:]
        #print("expr shape", shape(expr))
        expr=np.asarray(expr)
        #expr=expr[0:43261,0:2000] ###need remove
        #print("expr shape", shape(expr))
        expr = expr.transpose()
        #print("expr shape",shape(expr))
        self.cov_matrix = np.cov(expr)
        self.corr_matrix = np.corrcoef(expr)

    # new for time-series data
    def get_x_for_one_pair_version11_time(self, geneA, geneB):
        x = self.get_histogram_bins_time(geneA, geneB)
        histogram_list=[] 
        histogram_list=self.get_top_cov_pairs(geneA,geneB,"cov") # gene A-A,B-B,A-top_covA,B-top_covB             
        if len(histogram_list)>0:   
            #print("len histogram", len(histogram_list))       
            # concantate together, if ij not compress, consider the way put together. or consider multiple channel
            multi_image = self.cat_multiple_channel_zhang(x, histogram_list)
        else:
            multi_image = x         
        return multi_image
        
    def cat_multiple_channel_zhang(self, x_ij, histogram_list):
        x=[]
        x.append(x_ij)
        for i in range(0, len(histogram_list)):
            if histogram_list[i] is not None:
                x.append(histogram_list[i])
        #print("x shape",shape(x))
        return x

    def setting_for_one_pair(self, top_num=1000, add_self_image=True, get_abs=False,n_timepoints=1):

        self.top_num = top_num
        self.add_self_image = add_self_image
        self.get_abs = get_abs
        self.timepoints  = n_timepoints

        # print information after set some 
        print('\n')
        print('--------Setting info--------')
        print('num of top neighbor gene:', self.top_num)
        print('add A->A, B->B:',self.add_self_image)
        print('abs?:',self.get_abs)
        print('n_timepoints:',self.timepoints)
        print('split pairs based batch file?:',self.load_batch_split_pos)

        print('\n')
        
def main_for_representation_single_cell_type(out_dir, expr_file, pairs_for_predict_file, TF_divide_pos_file=None, geneName_map_file=None, TF_num=None,
                                             TF_pairs_num_lower_bound=0, TF_pairs_num_upper_bound=None,
                                             flag_load_split_batch_pos=True, add_self_image=True,
                                             get_abs=False,n_timepoints=1):
    if out_dir.endswith("/"):
        pass
    else:
        out_dir=out_dir+"/"

    ec = RepresentationTest2(out_dir, x_method_version=11, load_batch_split_pos=flag_load_split_batch_pos, start_batch_num=0, end_batch_num=TF_num, max_col=1)
        
    ec.setting_for_one_pair(top_num=10, add_self_image=add_self_image, get_abs=get_abs,n_timepoints=n_timepoints) # yu: top_num: num of top p-cov genes with gene A/B

    if flag_load_split_batch_pos:
        ec.load_split_batch_pos(TF_divide_pos_file)
    else:
        print('pls set it to be true and provide TF_divide_pos_file')

    ec.get_gene_list(geneName_map_file) 
  
    ec.load_real_data(expr_file,n_timepoints)  # YU: only load h5 expression files in the experiments

    ec.get_gold_standard(pairs_for_predict_file) # YU: get self.gold_standard and self.key_list

    ec.get_train_test(generate_multi=True,TF_pairs_num_lower_bound=TF_pairs_num_lower_bound,TF_pairs_num_upper_bound=TF_pairs_num_upper_bound)



if __name__ == '__main__':

    flag_load_split_batch_pos = (args.flag_load_split_batch_pos=='True')
  
    if args.TF_num=='None':
        TF_num = None
    else:
        TF_num = args.TF_num

    main_for_representation_single_cell_type(out_dir=args.out_dir, expr_file=args.expr_file, pairs_for_predict_file=args.pairs_for_predict_file, TF_divide_pos_file=args.TF_divide_pos_file, geneName_map_file=args.geneName_map_file, TF_num=TF_num,
                                             flag_load_split_batch_pos=flag_load_split_batch_pos,n_timepoints=args.n_timepoints)

