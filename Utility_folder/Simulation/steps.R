library(dyngen)

# backbone <- backbone_bifurcating_converging() # 10 time points, num_targets = 7900, num_hks = 2000,
# backbone <- backbone_bifurcating_cycle() # 6 time points, num_targets = 5900,num_hks = 4000,
# backbone <- backbone_bifurcating_loop() # 8 time points, num_targets = 6900,num_hks = 3000,
backbone <- backbone_consecutive_bifurcating() # 4 time points, num_targets = 4900,num_hks = 5000,


set.seed(1)

config <-
  initialise_model(
    backbone = backbone,
    num_cells = 20000,
    num_tfs = 100,
    num_targets = 4900,
    num_hks = 5000,
    verbose = FALSE,
	feature_network_params = feature_network_default(max_in_degree=1000),
	gold_standard_params = gold_standard_default(
		tau = 30/3600,
		census_interval = 4),
    simulation_params = simulation_default(
		total_time = 1000,
		census_interval = 4, 
		ssa_algorithm = ssa_etl(tau = 30/3600),
		experiment_params = simulation_type_wild_type(num_simulations = 100),
		compute_dimred = FALSE
    ),
	num_cores = 20, #getOption("Ncpus") %||% 1L,
	experiment_params = experiment_synchronised(num_timepoints = 4, pct_between = 0.1),
)	

save_dir <- '/tmp/csyuxu/dyngen_out'


out_tf_network <- generate_tf_network(config) # TFs and their edges
saveRDS(out_tf_network, file = paste(save_dir,"TF_net_info.rds",sep='/'))


out_f_network <- generate_feature_network(out_tf_network) # add target genes and housekeeping genes
saveRDS(out_f_network, file = paste(save_dir,"Feature_net_info.rds",sep='/'))


out_f_network_kinetics <- generate_kinetics(out_f_network) # add TFs -> targets edges / hk genes -> hk genes
saveRDS(out_f_network_kinetics, file = paste(save_dir,"After_kinetics_net_info.rds",sep='/'))


out_gs_simu <- generate_gold_standard(out_f_network_kinetics) # refer to the real GRN for gold standard simulation
saveRDS(out_gs_simu, file = paste(save_dir,"After_gold_standard_net_info.rds",sep='/'))
grn_info <- as.matrix(out_gs_simu$feature_network)
write.table(grn_info, file= paste(save_dir,"grn_info.txt",sep='/'), quote=FALSE, row.names=FALSE)


out_cell_simu <- generate_cells(out_gs_simu) # cell simulation
saveRDS(out_cell_simu, file = paste(save_dir,"After_cell_simulation_net_info.rds",sep='/'))


out_scexper_simu <- generate_experiment(out_cell_simu) # sample cell from different timepoints
saveRDS(out_cell_simu, file = paste(save_dir, "After_sc_experiment_net_info.rds",sep='/'))
cell_data <- out_scexper_simu$experiment$cell_info
cell_data <- as.matrix(cell_data)
write.table(cell_data, file= paste(save_dir,"cell_info.txt",sep='/'), quote=FALSE, row.names=FALSE)
#str(out_scexper_simu$experiment$cell_info)


scDataObject <- as_sce(out_scexper_simu)
counts <- scDataObject@assays@data@listData$counts
counts <- as.matrix(counts)
str(counts)
write.table(counts, file= paste(save_dir,"counts.txt",sep='/'), quote=FALSE)

#str(scDataObject@assays@data@listData$counts)

