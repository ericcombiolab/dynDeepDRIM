## Functional annotation assignment
### generate_input_funcassign.py

`python3 generate_input_funcassign.py -genepairs_filepath /home/comp/jxchen/xuyu/dynDeepDRIM/DB_func_annotation/immune_gene_pairs_train.txt -expression_filepath /home/comp/jxchen/xuyu/data/mouse_cortex -func_geneset_filepath /home/comp/jxchen/xuyu/dynDeepDRIM/DB_func_annotation/immune_known_gene.npy -n_timepoints 3 -save_dir /home/comp/jxchen/xuyu/dynDeepDRIM/func_annotation_input_tensors -save_filename immune_train`
`python3 generate_input_funcassign.py -genepairs_filepath /home/comp/jxchen/xuyu/dynDeepDRIM/DB_func_annotation/immune_gene_pairs_test.txt -expression_filepath /home/comp/jxchen/xuyu/data/mouse_cortex -func_geneset_filepath /home/comp/jxchen/xuyu/dynDeepDRIM/DB_func_annotation/immune_known_gene.npy -n_timepoints 3 -save_dir /home/comp/jxchen/xuyu/dynDeepDRIM/func_annotation_input_tensors -save_filename immune_test`
> -genepairs_filepath  
> -expression_filepath  
> -func_geneset_filepath  
> -n_timepoints  
> -save_dir  
> -save_filename

### dynDeepDRIM_func_annotation.py
`python3 dynDeepDRIM_func_annotation.py -train_data_path /home/comp/jxchen/xuyu/dynDeepDRIM/func_annotation_input_tensors/immune_train.npy -test_data_path /home/comp/jxchen/xuyu/dynDeepDRIM/func_annotation_input_tensors/immune_test.npy -output_dir Result_annotation -count_set_path /home/comp/jxchen/xuyu/dynDeepDRIM/DB_func_annotation/cell_cycle_count_set_test.txt -annotation_name cell_cycle`
> -train_data_path  
> -test_data_path  
> -output_dir  
> -count_set_path  
> -annotation_name  


## TF-gene prediction for GRN reconstruction
### generate_input_tfgene.py
`python3 generate_input_tfgene.py -out_dir mesc2_representation -expr_file /home/comp/jxchen/xuyu/data/mesc2_expression_data -pairs_for_predict_file /home/comp/jxchen/xuyu/dynDeepDRIM/DB_pairs_TF_gene/mesc2_gene_pairs_400.txt -geneName_map_file /home/comp/jxchen/xuyu/dynDeepDRIM/DB_pairs_TF_gene/mesc2_gene_list_ref.txt -TF_divide_pos_file /home/comp/jxchen/xuyu/dynDeepDRIM/DB_pairs_TF_gene/mesc2_gene_pairs_400_num.txt -TF_num 38 -n_timepoints 4 -flag_load_split_batch_pos True`
> -out_dir  
> -expr_file  
> -pairs_for_predict_file  
> -geneName_map_file  
> -flag_load_split_batch_pos  
> -TF_divide_pos_file
> -TF_num  
> -n_timepoints

### dynDeepDRIM_TF_gene.py
`python3 dynDeepDRIM_TF_gene.py -num_batches 36 -data_path /home/comp/jxchen/xuyu/DRIM_T/hesc1_representation/v_DT/ -output_dir hesc1_TFpred -cross_validation_fold_divide_file /home/comp/jxchen/xuyu/dynDeepDRIM/DB_pairs_TF_gene/hesc1_cross_validation_fold_divide.txt`
> -num_batches  
> -data_path
> -output_dir  
> -cross_validation_fold_divide_file
