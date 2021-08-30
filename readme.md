# dynDeepDRIM: a dynamic deep learning model to infer direct regulatory interactions using single cell time-course gene expression data
***
## Single-cell time-course RNA-seq data
These data were normalized into RPKM for dynDeepDRIM from the raw read counts:  

https://zenodo.org/record/5335938#.YSxikY4zb4Y  

## Benchmark
Benchmark and gene pairs generation for mESC and hESC are avaliable from https://github.com/xiaoyeye/TDL .  

For gene annotation assignment, we downloaded the known gene set from [GSEA](https://www.gsea-msigdb.org/gsea/index.jsp) to evaluate the performance of dynDeepDRIM.  

***
## Code Environment
Code is tested using Python 3.6 and R 3.6.
***
## Task1: TF-gene prediction for GRN reconstruction
### Step1: Generating input tensor for dynDeepDRIM
#### Python script: generate_input_tfgene.py
Example of the command line:
``` bash
 python3 generate_input_tfgene.py -out_dir hesc1_representation -expr_file /home/comp/jxchen/xuyu/data/hesc1_expression_data -pairs_for_predict_file /home/comp/jxchen/xuyu/dynDeepDRIM/DB_pairs_TF_gene/hesc1_gene_pairs_400.txt -geneName_map_file /home/comp/jxchen/xuyu/dynDeepDRIM/DB_pairs_TF_gene/hesc1_gene_list_ref.txt -TF_divide_pos_file /home/comp/jxchen/xuyu/dynDeepDRIM/DB_pairs_TF_gene/hesc1_gene_pairs_400_num.txt -TF_num 36 -n_timepoints 5
```
* -out_dir: Indicate the path for output.  
* -expr_file: The file of the normalized gene expression profile, which the row represents cell and the column represents gene.  
* -pairs_for_predict_file: The file of the training gene pairs and their labels.  
* -geneName_map_file: The file to map the name of gene in expr_file to the pairs_for_predict_file.   
* -TF_divide_pos_file: File that indicate the position in pairs_for_predict_file to divide pairs into different TFs.
* -TF_num: To generate representation for this number of TFs. Should be a integer that equal or samller than the number of TFs in the pairs_for_predict_file.  
* -n_timepoints: The number of time points in time-course dataset.

### Step2: Train and test the model
#### Python script: dynDeepDRIM_TF_gene.py
Example of the command line:
``` bash
python3 dynDeepDRIM_TF_gene.py -num_batches 36 -data_path /home/comp/jxchen/xuyu/dynDeepDRIM/hesc1_representation/v_dynDeepDRIM/ -output_dir hesc1_TFpred -cross_validation_fold_divide_file /home/comp/jxchen/xuyu/dynDeepDRIM/DB_pairs_TF_gene/hesc1_cross_validation_fold_divide.txt
```
* -num_batches: Since in STEP 1, we divide training pairs by TFs, and representation for one TF is included in one batch. Here the num_batches should be the number of TF.  
* -data_path: The path that includes x file, y file and z file, which is generated in the last step.
* -output_dir: Indicate the path for output.  
-cross_validation_fold_divide_file: A file that indicate how to divide the x file into three-fold. 


## Task2: Functional annotation assignment of genes
### Input tensor generation: generate_input_funcassign.py
+ Generateing train set
> python3 generate_input_funcassign.py -genepairs_filepath /home/comp/jxchen/xuyu/dynDeepDRIM/DB_func_annotation/immune_gene_pairs_train.txt -expression_filepath /home/comp/jxchen/xuyu/data/mouse_cortex -func_geneset_filepath /home/comp/jxchen/xuyu/dynDeepDRIM/DB_func_annotation/immune_known_gene.npy -n_timepoints 3 -save_dir /home/comp/jxchen/xuyu/dynDeepDRIM/func_annotation_input_tensors -save_filename immune_train
+ Generateing test set
>python3 generate_input_funcassign.py -genepairs_filepath /home/comp/jxchen/xuyu/dynDeepDRIM/DB_func_annotation/immune_gene_pairs_test.txt -expression_filepath /home/comp/jxchen/xuyu/data/mouse_cortex -func_geneset_filepath /home/comp/jxchen/xuyu/dynDeepDRIM/DB_func_annotation/immune_known_gene.npy -n_timepoints 3 -save_dir /home/comp/jxchen/xuyu/dynDeepDRIM/func_annotation_input_tensors -save_filename immune_test
>> -genepairs_filepath  
>> -expression_filepath  
>> -func_geneset_filepath  
>> -n_timepoints  
>> -save_dir  
>> -save_filename

### Train and test the model: dynDeepDRIM_func_annotation.py
> python3 dynDeepDRIM_func_annotation.py -train_data_path /home/comp/jxchen/xuyu/dynDeepDRIM/func_annotation_input_tensors/immune_train.npy -test_data_path /home/comp/jxchen/xuyu/dynDeepDRIM/func_annotation_input_tensors/immune_test.npy -output_dir Result_annotation -count_set_path /home/comp/jxchen/xuyu/dynDeepDRIM/DB_func_annotation/immune_count_set_test.txt -annotation_name immune
>> -train_data_path  
>> -test_data_path  
>> -output_dir  
>> -count_set_path  
>> -annotation_name  

