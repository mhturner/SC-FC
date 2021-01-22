library(nat.flybrains)
library(nat.jrcbrains)
library(nat.h5reg)
library(neuprintr)
library(dplyr)
library(bioimagetools)

# data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'
data_dir = '/oak/stanford/groups/trc/data/Max/flynet/data'

t0 = Sys.time()

# nat.jrcbrains:::list_saalfeldlab_registrations()
# plot(bridging_graph(), vertex.size=15, edge.arrow.size=.5)

# Load neuron / body IDs
body_ids =  read.csv(file.path(data_dir, 'connectome_connectivity', 'body_ids.csv'), header = FALSE)
# body_ids = sample_n(body_ids, 100)

# Load atlas
res = 0.38 # um/voxel of atlas
suppressWarnings({
  branson_atlas <- bioimagetools::readTIF(file.path(data_dir, '2018_999_atlas.tif'), as.is=TRUE)
})
count_matrix <- matrix(0, max(branson_atlas), max(branson_atlas))
syn_mask <- array(0, dim=dim(branson_atlas))
for (ind in 1:dim(body_ids)[1]){
  body_id = body_ids[ind,1]
  syn_data = neuprint_get_synapses(body_id, replace=FALSE)
  
  input_hemi = as.data.frame(syn_data[syn_data$prepost==1, c("x", "y", "z")]) * 8/1000 # vox -> um
  output_hemi = as.data.frame(syn_data[syn_data$prepost==0, c("x", "y", "z")]) * 8/1000 # vox -> um
  
  # Go from hemibrain space to JRC2018F (microns), then convert to JRC2018F voxels
  input_jrc2018f = data.matrix(xform_brain(input_hemi, sample = JRCFIB2018F, reference = JRC2018F) / res) # um -> atlas voxels
  output_jrc2018f = data.matrix(xform_brain(output_hemi, sample = JRCFIB2018F, reference = JRC2018F) / res) # um -> atlas voxels
  
  # Swap x and y for indexing
  input_jrc2018f = input_jrc2018f[ , c("y", "x", "z")]
  output_jrc2018f = output_jrc2018f[ , c("y", "x", "z")]
  
  mode(input_jrc2018f) = 'integer' # floor to int to index
  input_regions = unique(branson_atlas[input_jrc2018f]) # subset atlas matrix with integer array
  input_regions = input_regions[input_regions!=0] # remove 0 regions (non-brain)
  
  mode(output_jrc2018f) = 'integer' # floor to int to index
  output_regions = unique(branson_atlas[output_jrc2018f]) # subset atlas matrix with integer array
  output_regions = output_regions[output_regions!=0]
  
  syn_mask[input_jrc2018f] = syn_mask[output_jrc2018f] + 1 # number of outputting cells in each voxel of atlas space (doesn't count multiple t-bars in a voxel for one cell)
  count_matrix[input_regions, output_regions] = count_matrix[input_regions, output_regions] + 1
}

write.csv(count_matrix, file.path(data_dir, 'JRC2018F_branson_cellcount_matrix.csv'))
writeTIF(syn_mask, file.path(data_dir, 'JRC2018F_branson_synmask.tif'))

Sys.time() - t0