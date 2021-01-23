library(nat.flybrains)
library(nat.jrcbrains)
library(nat.h5reg)
library(neuprintr)
library(dplyr)
library(bioimagetools)

comparison_space <- commandArgs(trailingOnly = TRUE)

# data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'
data_dir = '/oak/stanford/groups/trc/data/Max/flynet/data'

t0 = Sys.time()

# Load neuron / body IDs
body_ids =  read.csv(file.path(data_dir, 'connectome_connectivity', 'body_ids.csv'), header = FALSE)
body_ids = sample_n(body_ids, 100) # testing

# Load atlas(es)
if (comparison_space == 'JFRC2'){
  res = 0.68 # um/voxel of atlas
  suppressWarnings({
    ito_atlas <- bioimagetools::readTIF(file.path(data_dir, 'template_brains', 'JFRCtempate2010.mask130819_Original.tif'), as.is=TRUE)
    branson_atlas <- bioimagetools::readTIF(file.path(data_dir, 'template_brains', 'AnatomySubCompartments20150108_ms999centers.tif'), as.is=TRUE)
  })
  
} else if (comparison_space == 'JRC2018'){
  res = 0.38 # um/voxel of atlas
  suppressWarnings({
    ito_atlas <- bioimagetools::readTIF(file.path(data_dir, 'template_brains', 'ito_2018.tif'), as.is=TRUE)
    branson_atlas <- bioimagetools::readTIF(file.path(data_dir, 'template_brains', '2018_999_atlas.tif'), as.is=TRUE)
  })
  
} else {
  print('Unrecognized comparison space argument')
}

branson_count_matrix <- matrix(0, max(branson_atlas), max(branson_atlas))
branson_tbar_matrix <- matrix(0, max(branson_atlas), max(branson_atlas))
branson_weighted_tbar_matrix <- matrix(0, max(branson_atlas), max(branson_atlas))

ito_count_matrix <- matrix(0, max(ito_atlas), max(ito_atlas))
ito_tbar_matrix <- matrix(0, max(ito_atlas), max(ito_atlas))
ito_weighted_tbar_matrix <- matrix(0, max(branson_atlas), max(branson_atlas))

syn_mask <- array(0, dim=dim(ito_atlas))

# get synapses associated with bodies
syn_data = neuprint_get_synapses(body_ids[,1])

# convert hemibrain raw locations to microns
syn_data[,c("x", "y", "z")] = syn_data[,c("x", "y", "z")] * 8/1000 # vox -> um

# Go from hemibrain space to comparison space
if (comparison_space == 'JFRC2'){
  syn_data[,c("x", "y", "z")] = xform_brain(syn_data[,c("x", "y", "z")], sample = JRCFIB2018F, reference = JFRC2) / res # x,y,z um -> atlas voxels
} else if (comparison_space == 'JRC2018') {
  syn_data[,c("x", "y", "z")] = xform_brain(syn_data[,c("x", "y", "z")], sample = JRCFIB2018F, reference = JRC2018F) / res # x,y,z um -> atlas voxels
}

# split into input / output 
input_synapses = as.data.frame(syn_data[syn_data$prepost==1, c("x", "y", "z", "bodyid")])
output_synapses = as.data.frame(syn_data[syn_data$prepost==0, c("x", "y", "z", "bodyid")])

# For each cell in synapse list
for (body_id in body_ids[,1]){
  # Swap x and y for indexing
  input_yxz = data.matrix(input_synapses[input_synapses$bodyid==body_id, c("y", "x", "z")])
  output_yxz = data.matrix(output_synapses[output_synapses$bodyid==body_id, c("y", "x", "z")])
  
  mode(input_yxz) = 'integer' # floor to int to index
  mode(output_yxz) = 'integer' # floor to int to index
  
  # # # # # # BRANSON ATLAS # # # # # # # # # # # #
  input_regions = branson_atlas[input_yxz] # subset atlas matrix with integer array
  input_regions = input_regions[input_regions!=0] # remove 0 regions (non-brain)
  input_tab = table(input_regions)
  input_regions = as.numeric(names(input_tab)) # now unique regions
  input_counts = as.vector(input_tab)
  
  output_regions = branson_atlas[output_yxz]
  output_regions = output_regions[output_regions!=0]
  output_tab = table(output_regions) 
  output_regions = as.numeric(names(output_tab))  # now unique regions
  output_counts = as.vector(output_tab)

  # Cell count 
  branson_count_matrix[input_regions, output_regions] = 
    branson_count_matrix[input_regions, output_regions] + 1
  
  # Total T-bar count 
  branson_tbar_matrix[input_regions, output_regions] = 
    branson_tbar_matrix[input_regions, output_regions] + 
    t(replicate(length(input_regions), output_counts))
  
  # Weighted T-bar count: output tbars mult. by fraction of total input synapses in source region 
  branson_weighted_tbar_matrix[input_regions, output_regions] = 
    branson_weighted_tbar_matrix[input_regions, output_regions] + 
    as.matrix(input_counts / sum(input_counts)) %*% t(as.matrix(output_counts))
    
  # # # # # # ITO ATLAS # # # # # # # # # # # #
  input_regions = ito_atlas[input_yxz] # subset atlas matrix with integer array
  input_regions = input_regions[input_regions!=0] # remove 0 regions (non-brain)
  input_tab = table(input_regions)
  input_regions = as.numeric(names(input_tab)) # now unique regions
  input_counts = as.vector(input_tab)
  
  output_regions = ito_atlas[output_yxz]
  output_regions = output_regions[output_regions!=0]
  output_tab = table(output_regions) 
  output_regions = as.numeric(names(output_tab))  # now unique regions
  output_counts = as.vector(output_tab)
  
  # Cell count 
  ito_count_matrix[input_regions, output_regions] = 
    ito_count_matrix[input_regions, output_regions] + 1
  
  # Total T-bar count 
  ito_tbar_matrix[input_regions, output_regions] = 
    ito_tbar_matrix[input_regions, output_regions] + 
    t(replicate(length(input_regions), output_counts))
  
  # Weighted T-bar count: output tbars mult. by fraction of total input synapses in source region 
  ito_weighted_tbar_matrix[input_regions, output_regions] = 
    ito_weighted_tbar_matrix[input_regions, output_regions] + 
    as.matrix(input_counts / sum(input_counts)) %*% t(as.matrix(output_counts))

  
  # Append output synapse counts to synapse mask
  ct_by_vox = aggregate(data.frame(output_yxz)$x, by=data.frame(output_yxz), length)
  syn_mask[data.matrix(ct_by_vox)[,1:3]] = syn_mask[data.matrix(ct_by_vox)[,1:3]] + data.matrix(ct_by_vox)[,4]
  
}

# Save conn matrices and syn mask
write.csv(branson_count_matrix, file.path(data_dir, 'hemi_2_atlas', paste(comparison_space, 'branson_cellcount_matrix.csv', sep='_')))
write.csv(branson_tbar_matrix, file.path(data_dir, 'hemi_2_atlas', paste(comparison_space, 'branson_tbar_matrix', sep='_')))
write.csv(branson_weighted_tbar_matrix, file.path(data_dir, 'hemi_2_atlas', paste(comparison_space, 'branson_weighted_tbar_matrix', sep='_')))

write.csv(ito_count_matrix, file.path(data_dir, 'hemi_2_atlas', paste(comparison_space, 'ito_cellcount_matrix.csv', sep='_')))
write.csv(ito_tbar_matrix, file.path(data_dir, 'hemi_2_atlas', paste(comparison_space, 'ito_tbar_matrix', sep='_')))
write.csv(ito_weighted_tbar_matrix, file.path(data_dir, 'hemi_2_atlas', paste(comparison_space, 'ito_weighted_tbar_matrix', sep='_')))

print(max(syn_mask))
writeTIF((2 ** 16) * syn_mask / max(syn_mask), file.path(data_dir, 'hemi_2_atlas', paste(comparison_space, 'synmask.tif', sep='_')))

Sys.time() - t0