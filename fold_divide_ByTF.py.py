import numpy as np
import os
import argparse

n_fold = 3

parser = argparse.ArgumentParser(description="")
parser.add_argument('-out_filename', required=True, help='output file name')
parser.add_argument('-pos_filepath', required=True, help='TF pos file path')

args = parser.parse_args()

## read pos file
s = open(args.pos_filepath,'r')
pos_list = []
for line in s:
    read = line.split()[0]
    pos_list.append(read)
s.close()
    
for i in range(len(pos_list)):
    pos_list[i] = int(pos_list[i])

## trans pos information to n_sample
num_list = []
for i in range(len(pos_list)-1):
    num_list.append(pos_list[i+1]-pos_list[i])

num_list_sort = num_list
num_list_sort.sort(reverse=True)

## average group(fold)
balance_groups = [[] for i in range(n_fold)]
for v in num_list_sort:
    balance_groups.sort(key=lambda x: sum(x))
    balance_groups[0].append(v)
 
balance_groups_index = [[] for i in range(n_fold)] 
for i in range(n_fold):
    balance_groups_index[i] = np.where(np.in1d(num_list,balance_groups[i]))[0].tolist()

## save TF index    
save_dir = os.getcwd()
#save_file = open(save_dir+'\\'+ args.out_filename + '.txt','w') # windows
save_file = open(save_dir+'/'+ args.out_filename + '.txt','w') # linux

for i in range(n_fold):
    for j in range(len(balance_groups_index[i])):
        save_file.write(str(balance_groups_index[i][j]))
        if j < len(balance_groups_index[i])-1:
            save_file.write(',')
    save_file.write('\n')
save_file.close()
