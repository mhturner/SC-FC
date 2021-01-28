library(neuprintr)
library(nat)
library(dplyr)

data_dir = '/home/mhturner/Dropbox/ClandininLab/Analysis/SC-FC/data'

# Colours
## some nice colors!! Inspired by LaCroixColoR
lacroix = c("#C70E7B", "#FC6882", "#007BC3", "#54BCD1", "#EF7C12", "#F4B95A", 
            "#009F3F", "#8FDA04", "#AF6125", "#F4E3C7", "#B25D91", "#EFC7E6", 
            "#EF7C12", "#F4B95A", "#C23A4B", "#FBBB48", "#EFEF46", "#31D64D", 
            "#132157","#EE4244", "#D72000", "#1BB6AF")
names(lacroix) = c("purple", "pink",
                   "blue", "cyan",
                   "darkorange", "paleorange",
                   "darkgreen", "green",
                   "brown", "palebrown",
                   "mauve", "lightpink",
                   "orange", "midorange",
                   "darkred", "darkyellow",
                   "yellow", "palegreen", 
                   "navy","cerise",
                   "red", "marine")
#  eg1: LNO --------------------------------------------------------------------
neur = neuprint_search("LNO.*", field = "type", meta=TRUE)
body_ids = neuprint_ids(neur$bodyid)
skels = neuprint_read_skeletons(body_ids)

plot3d(skels, col = sample(lacroix,length(skels), replace = TRUE), lwd = 2)

NO.mesh = neuprint_ROI_mesh(roi = "NO")
plot3d(NO.mesh, add = TRUE, alpha = 0.5, col = )
LAL.mesh = neuprint_ROI_mesh(roi = "LAL(R)")
plot3d(LAL.mesh, add = TRUE, alpha = 0.5, col = "grey")
AL.mesh = neuprint_ROI_mesh(roi = "AL(R)")
plot3d(AL.mesh, add = TRUE, alpha = 0.5, col = "grey")

um = matrix(c(0.8894074, -0.2930942, -0.3507853, 0, 0.1790030, 0.9294183, -0.3227063, 0, 0.4206095, 0.2242260, 0.8790962, 0, 0.0000000, 0.0000000, 0.0000000, 1), nrow=4, ncol=4, byrow=TRUE)
view3d(userMatrix = um)

rgl.snapshot(filename = file.path(data_dir, 'hemi_2_atlas', 'LNO_skels.png'), fmt ="png")

# %%

#  eg2:  ER--------------------------------------------------------------------

neur = neuprint_search("ER.*", field = "type", meta=TRUE)
body_ids = neuprint_ids(neur$bodyid)
skels = neuprint_read_skeletons(body_ids)

plot3d(skels, col = sample(lacroix,length(skels), replace = TRUE), lwd = 2)

EB.mesh = neuprint_ROI_mesh(roi = "EB")
plot3d(EB.mesh, add = TRUE, alpha = 0.5, col = lacroix[["orange"]])

rgl.snapshot(filename = file.path(data_dir, 'hemi_2_atlas', 'EBR_skels.png'), fmt ="png")

