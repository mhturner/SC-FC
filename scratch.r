body_ids =  read.csv(file.path(data_dir, 'connectome_connectivity', 'body_ids.csv'), header = FALSE)
body_ids = sample_n(body_ids, 10)


ct = 0
for (ind in 1:dim(body_ids)[1]){
  
  body_id = body_ids[ind,1]
 ct = ct + 1 
 print(body_id)

}