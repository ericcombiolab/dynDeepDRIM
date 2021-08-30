# dynDeepDRIM: a dynamic deep learning model to infer direct regulatory interactions using single cell time-course gene expression data

## Single-cell time-course RNA-seq data
These data were normalized into RPKM for this experiment from the raw read counts:  

https://zenodo.org/record/5335938#.YSxikY4zb4Y  

## Benchmark
Benchmark and gene pairs generation for mESC and hESC are avaliable from https://github.com/xiaoyeye/TDL .  

For gene annotation assignment, we downloaded the known gene set from [GSEA](https://www.gsea-msigdb.org/gsea/index.jsp) to evaluate the performance of dynDeepDRIM.  


## Code Environment
Code is tested using Python 3.6.8

## TASK1: TF-gene prediction for GRN reconstruction
### STEP1: Generating input tensor for dynDeepDRIM
#### Python script: generate_input_tfgene.py
Example of the command line:
``` bash
python3 generate_input_tfgene.py -out_dir hesc1_representation -expr_file /home/comp/jxchen/xuyu/data/hesc1_expression_data -pairs_for_predict_file /home/comp/jxchen/xuyu/dynDeepDRIM/DB_pairs_TF_gene/hesc1_gene_pairs_400.txt -geneName_map_file /home/comp/jxchen/xuyu/dynDeepDRIM/DB_pairs_TF_gene/hesc1_gene_list_ref.txt -TF_divide_pos_file /home/comp/jxchen/xuyu/dynDeepDRIM/DB_pairs_TF_gene/hesc1_gene_pairs_400_num.txt -TF_num 36 -n_timepoints 5
```
* -out_dir: Indicate the path for output.  
* -expr_file: The file of the normalized gene expression profile, which the row represents cell and the column represents gene. The preprocessed expression data(RPKM) are available in [Here](https://zenodo.org/record/5335938#.YSxikY4zb4Y)   
* -pairs_for_predict_file: The file of the training gene pairs and their labels. The files used in this experiment are available in folder [DB_pairs_TF_gene](https://github.com/yuxu-1/dynDeepDRIM/tree/master/DB_pairs_TF_gene), for example, [hesc1_gene_pairs_400.txt](https://github.com/yuxu-1/dynDeepDRIM/blob/master/DB_pairs_TF_gene/hesc1_gene_pairs_400.txt). 
* -geneName_map_file: The file to map the name of gene in expr_file to the pairs_for_predict_file. For example, [hesc1_gene_list_ref.txt](https://github.com/yuxu-1/dynDeepDRIM/blob/master/DB_pairs_TF_gene/hesc1_gene_list_ref.txt).  
* -TF_divide_pos_file: File that indicate the position in pairs_for_predict_file to divide pairs into different TFs. For example, [hesc1_gene_pairs_400_num.txt](https://github.com/yuxu-1/dynDeepDRIM/blob/master/DB_pairs_TF_gene/hesc1_gene_pairs_400_num.txt).
* -TF_num: To generate representation for this number of TFs. Should be a integer that equal or samller than the number of TFs in the pairs_for_predict_file.  
* -n_timepoints: The number of time points in time-course dataset.

Example output:
+ x file: The representation of genes' expression file, use as the input of the model.
+ y file: The label for the corresponding pairs. 
+ z file: Indicate the gene name for each pair. 

### STEP2: Train and test the model
#### Python script: dynDeepDRIM_TF_gene.py
Example of the command line:
``` bash
python3 dynDeepDRIM_TF_gene.py -num_batches 36 -data_path /home/comp/jxchen/xuyu/dynDeepDRIM/hesc1_representation/v_dynDeepDRIM/ -output_dir hesc1_TFpred -cross_validation_fold_divide_file /home/comp/jxchen/xuyu/dynDeepDRIM/DB_pairs_TF_gene/hesc1_cross_validation_fold_divide.txt
```
* -num_batches: Since in STEP 1, we divide training pairs by TFs, and representation for one TF is included in one batch. Here the num_batches should be the number of TF.  
* -data_path: The path that includes x file, y file and z file, which is generated in the last step.
* -output_dir: Indicate the path for output.  
-cross_validation_fold_divide_file: A file that indicate how to divide the x file into three-fold. For example, [hesc1_cross_validation_fold_divide.txt](https://github.com/yuxu-1/dynDeepDRIM/blob/master/DB_pairs_TF_gene/hesc1_cross_validation_fold_divide.txt) 


## TASK2: Functional annotation assignment of genes
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

