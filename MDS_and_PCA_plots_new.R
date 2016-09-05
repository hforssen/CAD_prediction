library(reshape2)
library(ggplot2)


# ########################################################
# ##########  Non-standardised ########### 
# 
# data_miss <- read.csv('full_nonstandard_data_with_missing.csv')
# data_nomiss <- read.csv('nonstandard_data_without_missing.csv')
# 
# data_miss1 <- data_miss[,2:227]
# data_nomiss1 <- data_nomiss[,2:227]
# 
# 
# ######## Missing dataset (missings just excluded, so may not be as accurate) ######## 
# data_dist <- dist(data_miss1[1:225],method="euclidean")
# mds_data <- cmdscale(data_dist,k=2)
# 
# mds_dataframe <- as.data.frame(mds_data)
# 
# mds_dataframe[,"CAD50"] = data_miss1[,226]
# 
# plot(mds_dataframe[,1:2],xlab="MDS scaling 1", ylab="MDS scaling 2",col=ifelse(mds_dataframe[,"CAD50"] == 1,"orange", "blue"), pch =20)
# # no real pattern can be seen
# 
# 
# # Cannot do PCA with missing data!!!
# 
# 
# ######## Complete dataset ######## 
# data_dist_no <- dist(data_nomiss1[1:225],method="euclidean")
# mds_data_no <- cmdscale(data_dist_no,k=2)
# 
# mds_dataframe_no <- as.data.frame(mds_data_no)
# 
# mds_dataframe_no[,"CAD50"] = data_nomiss1[,226]
# 
# plot(mds_dataframe_no[,1:2],xlab="MDS scaling 1", ylab="MDS scaling 2",col=ifelse(mds_dataframe_no[,"CAD50"] == 1,"orange", "blue"), pch =20)
# # no real pattern can be seen
# 
# 
# 
# # Plot first 2 PCs against eachother
# pca_dataframe <- prcomp(data_nomiss1[,1:225],
#                  center = TRUE,
#                  scale. = TRUE) 
# scores <- data.frame(mds_dataframe_no, pca_dataframe$x[,1:2])
# pc1_2 <- qplot(x=PC1, y=PC2, data=scores, colour=factor(mds_dataframe_no[,"CAD50"]), size = I(3)) +
#   theme(legend.position="none")
# pc1_2
# 
# 
# ########################################################
# # Imputed data
# data_dist <- dist(imp_data5[1:225],method="euclidean")
# mds_data <- cmdscale(data_dist,k=2)
# mds_dataframe <- as.data.frame(mds_data)
# mds_dataframe[,"CAD50"] = data_miss1[,226]
# 
# plot(mds_dataframe[,1:2],xlab="MDS scaling 1", ylab="MDS scaling 2",col=ifelse(mds_dataframe[,"CAD50"] == 1,"red", "black"), pch =20, cex = 1.4)
# 


###########################################################################
##########  Standardised ########### 
###########################################################################

# Load data with missing, calculate mean and variance. Load imputed, average, standardise.
data_miss_full <- read.csv('full_nonstandard_data_with_missing.csv')
imp_data1_full <- read.csv('imp_data1_nonstand.csv')
imp_data2_full <- read.csv('imp_data2_nonstand.csv')
imp_data3_full <- read.csv('imp_data3_nonstand.csv')
imp_data4_full <- read.csv('imp_data4_nonstand.csv')
imp_data5_full <- read.csv('imp_data5_nonstand.csv')

data_miss1 <- data_miss_full[,2:227]
imp_data1 <- imp_data1_full[,2:227]
imp_data2 <- imp_data2_full[,2:227]
imp_data3 <- imp_data3_full[,2:227]
imp_data4 <- imp_data4_full[,2:227]
imp_data5 <- imp_data5_full[,2:227]

data_mean = colMeans(data_miss1[,1:225],na.rm=TRUE)
data_sd = rep(0,225)
for (i in c(1:225))
  data_sd[i] = sd(imp_data1[,i])


# Subtract means
imp_data1_sd = matrix(0,1474,225)
for (i in c(1:225))
  imp_data1_sd[,i] = (imp_data1[,i] - data_mean[i])/data_sd[i]

imp_data2_sd = matrix(0,1474,225)
for (i in c(1:225))
  imp_data2_sd[,i] = (imp_data2[,i] - data_mean[i])/data_sd[i]

imp_data3_sd = matrix(0,1474,225)
for (i in c(1:225))
  imp_data3_sd[,i] = (imp_data3[,i] - data_mean[i])/data_sd[i]

imp_data4_sd = matrix(0,1474,225)
for (i in c(1:225))
  imp_data4_sd[,i] = (imp_data4[,i] - data_mean[i])/data_sd[i]

imp_data5_sd = matrix(0,1474,225)
for (i in c(1:225))
  imp_data5_sd[,i] = (imp_data5[,i] - data_mean[i])/data_sd[i]


imp_data1_standard = as.data.frame(imp_data1_sd)
imp_data1_standard['CAD50'] = imp_data1['CAD50']

imp_data2_standard = as.data.frame(imp_data2_sd)
imp_data2_standard['CAD50'] = imp_data2['CAD50']

imp_data3_standard = as.data.frame(imp_data3_sd)
imp_data3_standard['CAD50'] = imp_data3['CAD50']

imp_data4_standard = as.data.frame(imp_data4_sd)
imp_data4_standard['CAD50'] = imp_data4['CAD50']

imp_data5_standard = as.data.frame(imp_data5_sd)
imp_data5_standard['CAD50'] = imp_data5['CAD50']

# Give column names back
for (i in c(1:225))
  colnames( imp_data1_standard )[i] <- colnames( data_miss1 )[i]
  colnames( imp_data2_standard )[i] <- colnames( data_miss1 )[i]
  colnames( imp_data3_standard )[i] <- colnames( data_miss1 )[i]
  colnames( imp_data4_standard )[i] <- colnames( data_miss1 )[i]
  colnames( imp_data5_standard )[i] <- colnames( data_miss1 )[i]


##### MDS plot ##### 

data_dist <- dist(imp_data1_standard[1:225],method="euclidean")
mds_data <- cmdscale(data_dist,k=2)

mds_dataframe <- as.data.frame(mds_data)

mds_dataframe[,"CAD50"] = imp_data1_standard[,226]

plot(mds_dataframe[,1:2],xlab="Multidimesnional scaling dimension 1", ylab="Multidimesnional scaling dimension 2",col=ifelse(mds_dataframe[,"CAD50"] == 1,"turquoise3", "coral1"), pch =20)
legend("topleft", c("CAD", "No CAD"), pch = 20, col = c("turquoise3", "coral1"),bty='n')

##### PCA plot #####

pca_dataframe <- prcomp(imp_data1_standard[,1:225],
                        center = TRUE,
                        scale. = TRUE) 
scores <- data.frame(imp_data1_standard, pca_dataframe$x[,1:2])
scores['CAD'] <- factor(imp_data1_standard[,"CAD50"])
pc1_2 <- qplot(x=PC1, y=PC2, data=scores, colour=CAD,guide=FALSE, 
               size = I(3)) + theme_classic() +labs(x="First Principle Component Score",y="Second Principle Component Score",size=20)
pc1_2



############## Predictive variables only ############## 
colnames( imp_data1_standard )
mika_ala_vars1 = imp_data1_standard[c("MUFA.FA", "Phe","FAw6","DHA","PUFA","CAD50")]

mika_ala_vars2 = imp_data1_standard[c("MUFA.FA", "Phe","FAw6","DHA","PUFA","Tyr","Lac", "bOHBut","MUFA", "FAw3", "PUFA.FA",
                                      "UnSat", "Gp", "ApoA1", "ApoB", "Serum.TG", "XL.VLDL.L", "L.VLDL.L","M.VLDL.L",
                                      "S.VLDL.L", "XS.VLDL.L", "IDL.L", "L.LDL.L", "M.LDL.L", "S.LDL.L", "L.HDL.L", "M.HDL.L",
                                      "HDL.D", "VLDL.C", "IDL.C", "LDL.C", "HDL.C", "HDL2.C", "CAD50")]

random_forest_top15 = imp_data1_standard[c("Crea", "IDL.TG_.", "Phe", "Alb", "Lac", "IDL.C_.", "Gln", "bOHBut", "His", "Ace", 
                                           "M.LDL.TG_.", "Ala", "L.LDL.TG_.", "LA.FA", "Val", "CAD50")]

L1_regression_top15 = imp_data1_standard[c("ApoB.ApoA1", "S.LDL.CE", "SFA", "XXL.VLDL.PL", "S.LDL.TG_.", "S.VLDL.TG", "FreeC", 
                                          "HDL.D", "PC", "M.LDL.PL", "S.HDL.CE", "M.VLDL.CE", "XS.VLDL.FC_.", "ApoA1", "DHA.FA",
                                          "CAD50")]
                                           

##### PCA plot #####

pca_dataframe <- prcomp(random_forest_top15[,1:5],
                        center = TRUE,
                        scale. = TRUE) 
scores <- data.frame(L1_regression_top15, pca_dataframe$x[,1:2])
pc1_2 <- qplot(x=PC1, y=PC2, data=scores, colour=factor(random_forest_top15[,"CAD50"]), size = I(1)) +
  theme(legend.position="none")
pc1_2



