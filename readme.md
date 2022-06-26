# dynDeepDRIM: a dynamic deep learning model to infer direct regulatory interactions using time-course single cell gene expression data

*******

## Time-course single-cell RNA-seq data
These data were normalized by [**sctransform**](https://github.com/satijalab/sctransform) from the raw read counts.    
Zenodo: The normalized datasets used in dynDeepDRIM can be downloaded from [**link**](https://zenodo.org/record/6720690#.YrXQjHZBz4Y) directly (version2).
#### Gene Expression Datasets for GRN reconstruction:  
* Four simulation datasets generated by [**dyngen**](https://github.com/dynverse/dyngen);   
* Four real ESC datasets. 
#### Gene Expression Datasets for gene functional annotation:  
* Mouse brain cortex. 

*******

## Benchmark
* Benchmark for the simulation datasets: [**link**](https://github.com/yuxu-1/dynDeepDRIM/tree/master/DB_pairs_TF_gene).
* Benchmark for mESC and hESC datasets are avaliable from [**link**](https://github.com/yuxu-1/dynDeepDRIM/tree/master/DB_pairs_TF_gene) (based on https://github.com/xiaoyeye/TDL).  
* Benchmark for gene functional annotation, the gene sets are avaliable from [**link**](https://github.com/yuxu-1/dynDeepDRIM/tree/master/DB_func_annotation), downloaded from [**Gene Ontology**](https://www.gsea-msigdb.org/gsea/index.jsp) (GO:0050890 (cognition); GO:0007600 (sensory perception; GO:0099536(synaptic signaling), and GO:0030424 (axon)) .

*******

## Code Environment
The file of conda environment installation is provided.
1. Download "environment.yaml" file in this repository;
2. Modify the path in the final line of "environment.yaml" file according to user's conda path. (e.g., `"/home/comp/csyuxu/anaconda3/envs/dynDeepDRIM"`.)
3. Run this command to install all python dependency for dynDeepDRIM
``` bash
conda env create -f environment.yaml
```
4. Activate conda env
``` bash
conda activate dynDeepDRIM
```

*******

## TASK1: TF-gene interactions inference for GRN reconstruction

### STEP1: Input tensors generation
#### Python script: generate_input_tfgene.py
Example of the command line:
``` bash
python generate_input_tfgene.py -out_dir /tmp/csyuxu/hesc1_input_tensors -expr_file /tmp/csyuxu/data/hesc1_expression_data -pairs_for_predict_file ./DB_pairs_TF_gene/hesc1_gene_pairs_400.txt -geneName_map_file ./DB_pairs_TF_gene/hesc1_gene_list_ref.txt -TF_divide_pos_file ./DB_pairs_TF_gene/hesc1_gene_pairs_400_num.txt -TF_num 36 -n_timepoints 5 -top_num 10 -image_resolution 8
```
* -out_dir: The path of output files.  
* -expr_file: The file of the normalized gene expression profile, which the row represents cell and the column represents gene.    
* -pairs_for_predict_file: The file of the gene pairs and their labels. e.g., [hesc1_gene_pairs_400.txt](https://github.com/yuxu-1/dynDeepDRIM/blob/master/DB_pairs_TF_gene/hesc1_gene_pairs_400.txt). 
* -geneName_map_file: The file to map the name of gene in expr_file to the pairs_for_predict_file. e.g., [hesc1_gene_list_ref.txt](https://github.com/yuxu-1/dynDeepDRIM/blob/master/DB_pairs_TF_gene/hesc1_gene_list_ref.txt).  
* -TF_divide_pos_file: The file that indicate the position in pairs_for_predict_file to divide pairs into different TFs. e.g., [hesc1_gene_pairs_400_num.txt](https://github.com/yuxu-1/dynDeepDRIM/blob/master/DB_pairs_TF_gene/hesc1_gene_pairs_400_num.txt).
* -TF_num: To generate representation for this number of TFs. Should be a integer that equal or samller than the number of TFs in the pairs_for_predict_file.  
* -n_timepoints: The number of time points in time-course dataset. Should be a integer that equal or samller than the number of time points in the dataset.
* -neighbor_criteria: The option of selecting correlation (corr) or covariance (cov) as measure for neighbor genes identification.
* -get_abs: Absolute the values of correlation/covariance in "neighbor_criteria"?
* -top_num: The number of neighbor genes to be involved, default 10.
* -image_resolution: default 8. 

Example output:
+ x file: The input tensors of dynDeepDRIM. Shape: (n_samples, 2*top_num+3, n_timepoints, image_resolution, image_resolution)
+ y file: The labels for the corresponding input tensors (pairs). 
+ z file: Indicate the gene name for each tensor (pair). 


### STEP2: Model evaluation by cross-validation
*Support n-fold cross-validation*
#### Python script: fold_divide_ByTF.py
Example of the command line:
``` bash
python fold_divide_ByTF.py -pos_filepath ./DB_pairs_TF_gene/hesc1_gene_pairs_400_num.txt -save_dir /home/comp/csyuxu/dynDeepDRIM -out_filename hesc1_cross_validation_fold_divide -n_folds 3
```
* -pos_filepath: The file that indicate the position in pairs_for_predict_file to divide pairs into different TFs. e.g., [hesc1_gene_pairs_400_num.txt](https://github.com/yuxu-1/dynDeepDRIM/blob/master/DB_pairs_TF_gene/hesc1_gene_pairs_400_num.txt).
* -save_dir: the path of output file.
* -out_filename: the name of output file.
* -n_folds: default 3

#### Python script: dynDeepDRIM_TF_gene.py
Example of the command line:
``` bash
CUDA_VISIBLE_DEVICES=2 python dynDeepDRIM_TF_gene.py -num_batches 36 -data_path /tmp/csyuxu/hesc1_input_tensors/v_dynDeepDRIM/ -output_dir tfgene_test -cross_validation_fold_divide_file ./DB_pairs_TF_gene/hesc1_cross_validation_fold_divide.txt
```
* -num_batches: Since in STEP 1, we divide training pairs by TFs, and representation for one TF is included in one batch. Here the num_batches should be the number of TF.  
* -data_path: The path of the input tensors, which is generated in STEP1.
* -output_dir: The path of output files.  
* -cross_validation_fold_divide_file: This file is generated by running `fold_divide_ByTF.py`. The information of n-folds (indexes represent the order of TFs in the gene pairs file [hesc1_gene_pairs_400.txt](https://github.com/yuxu-1/dynDeepDRIM/blob/master/DB_pairs_TF_gene/hesc1_gene_pairs_400.txt)). e.g., [hesc1_cross_validation_fold_divide.txt](https://github.com/yuxu-1/dynDeepDRIM/blob/master/DB_pairs_TF_gene/hesc1_cross_validation_fold_divide.txt) 

******

## TASK2: Gene functional annotataion
### STEP1: Generating input tensor for dynDeepDRIM 
#### python script: generate_input_funcassign.py
Example of the command line:
``` bash
python3 generate_input_funcassign.py -genepairs_filepath /home/comp/jxchen/xuyu/dynDeepDRIM/DB_func_annotation/immune_gene_pairs_train.txt -expression_filepath /home/comp/jxchen/xuyu/data/mouse_cortex -func_geneset_filepath /home/comp/jxchen/xuyu/dynDeepDRIM/DB_func_annotation/immune_known_gene.npy -n_timepoints 3 -save_dir /home/comp/jxchen/xuyu/dynDeepDRIM/func_annotation_input_tensors -save_filename immune_train

python3 generate_input_funcassign.py -genepairs_filepath /home/comp/jxchen/xuyu/dynDeepDRIM/DB_func_annotation/immune_gene_pairs_test.txt -expression_filepath /home/comp/jxchen/xuyu/data/mouse_cortex -func_geneset_filepath /home/comp/jxchen/xuyu/dynDeepDRIM/DB_func_annotation/immune_known_gene.npy -n_timepoints 3 -save_dir /home/comp/jxchen/xuyu/dynDeepDRIM/func_annotation_input_tensors -save_filename immune_test
```
+ -genepairs_filepath: The file of the training gene pairs and their labels. The files used in this experiment are available in folder [DB_func_annotation](https://github.com/yuxu-1/dynDeepDRIM/tree/master/DB_func_annotation), such as [immune](https://github.com/yuxu-1/dynDeepDRIM/blob/master/DB_func_annotation/immune_gene_pairs_train.txt).  
+ -expression_filepath: The file of the normalized gene expression profile, which the row represents cell and the column represents gene. The preprocessed expression data(RPKM) are available in [Here](https://zenodo.org/record/5335938#.YSxikY4zb4Y)   
+ -func_geneset_filepath: The shared genes between download gene set and genes in expression profiles. The neighbor genes will exclude these genes. The files used in this experiment are available in folder [DB_func_annotation](https://github.com/yuxu-1/dynDeepDRIM/tree/master/DB_func_annotation).
+ -n_timepoints: The number of time points in time-course dataset.  
+ -save_dir: Indicate the path for output.   
+ -save_filename: Suggests specific annotation name (e.g. immune_train).

Example output:  
+ immune_train.npy: The representation of genes' expression file, use as the input of the model for training.
+ immune_test.npy: The representation of genes' expression file, use as the input of the model for test.  

### STEP2: Train and test the model 
Before training the model, please make sure both training and test sets input tensors were generated.  
  
#### python script: dynDeepDRIM_func_annotation.py
Example of the command line:
``` bash
python3 dynDeepDRIM_func_annotation.py -train_data_path /home/comp/jxchen/xuyu/dynDeepDRIM/func_annotation_input_tensors/immune_train.npy -test_data_path /home/comp/jxchen/xuyu/dynDeepDRIM/func_annotation_input_tensors/immune_test.npy -output_dir Result_annotation -count_set_path /home/comp/jxchen/xuyu/dynDeepDRIM/DB_func_annotation/immune_count_set_test.txt -annotation_name immune
```
+ -train_data_path : The paht of the training input tensors generated in Step1.  
+ -test_data_path: The paht of the test input tensors generated in Step1.    
+ -output_dir: Indicate the path for output.     
+ -count_set_path: For calculating and collecting AUROCs of each g1(first gene in gene pairs) to assess the performance of the model. 
+ -annotation_name: The name of result folder 



## For static scRNA-seq data  
Refer to previous work by Jiaxing Chen, **DeepDRIM**: https://github.com/jiaxchen2-c/DeepDRIM


## Contact  
If any question, please feel free to contact with me or my supervisor Dr.Eric Zhang   
(Yu XU, email: csyuxu@comp.hkbu.edu.hk)  
(Lu Zhang, email: ericluzhang@hkbu.edu.hk) 

