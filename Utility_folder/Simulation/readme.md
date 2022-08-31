## Simulation datasets generation

### 1) Simulation by dyngen
1. GRN network setting
> Refer to dyngen's paper or tutorial for more details  

2. Modify the save path and then run the R script `steps.R`
``` bash
Rscript steps.R   
```
**Outputs: dyngen object**
+ TF_net_info.rds
+ Feature_net_info.rds
+ After_kinetics_net_info.rds
+ After_gold_standard_net_info.rds
+ After_cell_simulation_net_info.rds
+ After_sc_experiment_net_info.rds  
**Outputs: for time-course single-cell RNA-seq**
+ grn_info: TF-target binding pairs
+ cell_info: time points information of cells
+ counts: expression matrix (UMI)


### 2) Transitive interactions identification and benchmark gene pairs generation
1. Modify the save path and copy/move `grn_info.txt`, `cell_info.txt`, and `counts.txt` into save folder  
2. Run the python script `process_dyngen_results.py`
``` bash
python process_dyngen_results.py  
```
**Outputs: middle files**  
+ time_interval.txt
+ TFtarget_pairs.txt
+ non_ix_genepool.txt  
**Outputs: benchmark gene pairs for dynDeepDRIM**  
+ sorted_counts.txt: the expression matrix, in which the cells are grouped and then ordered by their time points. The number of cells in each time point is presented in the file of `time_interval.txt`
+ simulation_gene_list.txt: one of input files for dynDeepDRIM
+ simulation_gene_pairs.txt: one of input files for dynDeepDRIM
+ simulation_gene_pairs_num.txt: one of input files for dynDeepDRIM

