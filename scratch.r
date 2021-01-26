library(neuprintr)


neur = neuprint_search("LC[0-9].*", field = "type", meta=TRUE)

neur = neuprint_search("type:LC.*", meta=TRUE)


all_body_ids =  read.csv(file.path(data_dir, 'connectome_connectivity', 'body_ids.csv'), header = FALSE)
all_body_ids = sample_n(all_body_ids, 350) # testing
neuprint_ids()