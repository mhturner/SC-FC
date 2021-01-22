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
# body_ids = sample_n(body_ids, 10) # testing

# Load atlas(es)
if (comparison_space == 'JFRC2'){
  res = 0.68 # um/voxel of atlas
  suppressWarnings({
    ito_atlas <- bioimagetools::readTIF(file.path(data_dir, 'JFRCtempate2010.mask130819_Original.tif'), as.is=TRUE)
    branson_atlas <- bioimagetools::readTIF(file.path(data_dir, 'AnatomySubCompartments20150108_ms999centers.tif'), as.is=TRUE)
  })
  
} else if (comparison_space == 'JRC2018'){
  res = 0.38 # um/voxel of atlas
  suppressWarnings({
    ito_atlas <- bioimagetools::readTIF(file.path(data_dir, 'ito_2018.tif'), as.is=TRUE)
    branson_atlas <- bioimagetools::readTIF(file.path(data_dir, '2018_999_atlas.tif'), as.is=TRUE)
  })
  
} else {
  print('Unrecognized comparison space argument')
}

branson_count_matrix <- matrix(0, max(branson_atlas), max(branson_atlas))
ito_count_matrix <- matrix(0, max(ito_atlas), max(ito_atlas))
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
  
  # Find atlas regions that are input, output for this cell
  #   First - Branson atlas
  input_regions = unique(branson_atlas[input_yxz]) # subset atlas matrix with integer array
  input_regions = input_regions[input_regions!=0] # remove 0 regions (non-brain)
  
  output_regions = unique(branson_atlas[output_yxz]) # subset atlas matrix with integer array
  output_regions = output_regions[output_regions!=0]
  
  branson_count_matrix[input_regions, output_regions] = branson_count_matrix[input_regions, output_regions] + 1
  
  #   Second - Ito atlas
  input_regions = unique(ito_atlas[input_yxz]) # subset atlas matrix with integer array
  input_regions = input_regions[input_regions!=0] # remove 0 regions (non-brain)
  
  output_regions = unique(ito_atlas[output_yxz]) # subset atlas matrix with integer array
  output_regions = output_regions[output_regions!=0]
  
  ito_count_matrix[input_regions, output_regions] = ito_count_matrix[input_regions, output_regions] + 1
  
  # Append output synapse counts to synapse mask
  syn_mask[output_yxz] = syn_mask[output_yxz] + 1 # number of outputting cells in each voxel of atlas space (doesn't count multiple t-bars in a voxel for one cell)
  
}

write.csv(branson_count_matrix, file.path(data_dir, paste(comparison_space, 'branson_cellcount_matrix.csv', sep='_')))
write.csv(ito_count_matrix, file.path(data_dir, paste(comparison_space, 'ito_cellcount_matrix.csv', sep='_')))
writeTIF(syn_mask, file.path(data_dir, paste(comparison_space, 'synmask.tif', sep='_')))

Sys.time() - t0