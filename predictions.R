########################
# Dependencies
########################
install.packages('tidyr')
require(tidyr)
########################
# Reading Predicions
########################
tmp = read.csv("res.csv")
########################
# Converting Predictions to m * n Format
########################
res = tidyr::spread(tmp, 'Item', 'Rating', fill = 0)
res = as.data.frame(res)
########################
# Writing back to csv
########################
write.csv(res,"predictions.csv")
