# Get T-Bar locations for select cell types. To test alignment to atlas space.
# Turner, Mann, Clandinin.
# https://github.com/mhturner/SC-FC

library(nat.flybrains)
library(nat.jrcbrains)
library(nat.h5reg)
library(neuprintr)
library(dplyr)
library(bioimagetools)

options(warn=1)

# Input arg Cell type: LC, MBON, OPN, KC, ER, LNO (see below)
cell_type <- commandArgs(trailingOnly = TRUE)

data_dir = '/oak/stanford/groups/trc/data/Max/flynet/data'

# load atlas in JRC2018 space
res = 0.38 # um/voxel of atlas
ito_atlas <- bioimagetools::readTIF(file.path(data_dir, 'template_brains', 'ito_2018.tif'), as.is=TRUE)

# Craft neuprint cell search based on cell type
if (cell_type == 'LC'){
  neur = neuprint_search("LC[0-9].*", field = "type", meta=TRUE)
} else if (cell_type == 'MBON') {
  neur = neuprint_search("MBON[0-9][0-9]", field = "type", meta=TRUE)
} else if (cell_type == 'OPN') {
  neur = neuprint_search(".*vPN.*", field = "type", meta=TRUE)
} else if (cell_type == 'KC') {
  neur = neuprint_search("KCab-c", field = "type", meta=TRUE)
} else if (cell_type == 'ER') {
  neur = neuprint_search("ER.*", field = "type", meta=TRUE)
} else if (cell_type == 'LNO') {
  neur = neuprint_search("LNO.*", field = "type", meta=TRUE)
}

body_ids = neur$bodyid
# body_ids = body_ids[1:100] # testing
body_ids = neuprint_ids(body_ids)

# get synapses and types associated with bodies
syn_data = neuprint_get_synapses(body_ids)
syn_data = syn_data[!duplicated(syn_data[2:5]),] # remove duplicate T-bar locations (single t-bar -> multiple postsynapses)
types = neuprint_get_meta(body_ids)$type

# convert hemibrain raw locations to microns
syn_data[,c("x", "y", "z")] = syn_data[,c("x", "y", "z")] * 8/1000 # vox -> um
# Go from hemibrain space to JRC2018 space
syn_data[,c("x", "y", "z")] = xform_brain(syn_data[,c("x", "y", "z")], sample = JRCFIB2018F, reference = JRC2018F) / res # x,y,z um -> atlas voxels
# get output synapses (Tbars)
output_synapses = as.data.frame(syn_data[syn_data$prepost==0, c("x", "y", "z", "bodyid")])

cols = unique(types)
ito_tbar <- data.frame(matrix(0, ncol = length(cols), nrow = max(ito_atlas)))
colnames(ito_tbar) <- cols

for (i in seq_along(body_ids)){
  body_id = body_ids[i]
  type = types[i]
  output_yxz = data.matrix(output_synapses[output_synapses$bodyid==body_id, c("y", "x", "z")])
  mode(output_yxz) = 'integer' # floor to int to index
  
  # Find ito regions where tbars are
  output_regions = ito_atlas[output_yxz]
  output_regions = output_regions[output_regions!=0]
  output_tab = table(output_regions) 
  output_regions = as.numeric(names(output_tab))  # now unique regions
  output_counts = as.vector(output_tab)
  
  if (length(output_regions) > 0){
    ito_tbar[output_regions, type] = ito_tbar[output_regions, type] + output_counts
  }

} # end body_ids

write.csv(ito_tbar, file.path(data_dir, 'hemi_2_atlas', paste(cell_type, 'ito_tbar.csv', sep='_')))

