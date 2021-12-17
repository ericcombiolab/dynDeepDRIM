## Neigbor genes analysis

***

### Step1. Generate top $n$ neighbor genes file 
Run python script `find_neigbor_genes.py`
> python3 find_neigbor_genes.py -expr_dir /home/comp/csyuxu/dynDeepDRIM/data/mouse_cortex -positive_path /home/comp/csyuxu/dynDeepDRIM/DB_func_annotation/immune_known_gene.npy -negative_path /home/comp/csyuxu/dynDeepDRIM/DB_func_annotation/immune_unknown_gene.npy -func_name immune -func_name immune -top_n 10
* -expr_dir: the path of the normalized expression files (.h5 format)
* -positive_path: positive cases (genes) file. For example: [immune](https://github.com/yuxu-1/dynDeepDRIM/blob/master/DB_func_annotation/immune_known_gene.npy)
* -negative_path: positive cases (genes) file. For example: [immune](https://github.com/yuxu-1/dynDeepDRIM/blob/master/DB_func_annotation/immune_unknown_gene.npy)
* -func_name: the name of funtional genes
* -top_n: the number of top genes we would select, default=10  

Output: 
* posi_XXXX_neighborname.txt: genes in each row correspond with the gene in **-positive_path**
* nega_XXXX_neighborname.txt: genes in each row correspond with the gene in **-negative_path**

***

### Step2. Analyze the difference of neighbor context between positive and negative gene sets   
Running the `neighbor_genes_observation.ipynb` in jupyter notebook env

***

## Any question or recommendation
Please feel free to contact with me: csyuxu@comp.hkbu.edu.hk