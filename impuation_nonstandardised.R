### Imputation ### 

data_miss <- read.csv('full_nonstandard_data_with_missing.csv')

library(mice)

# Remove row ids
data_miss1 = data_miss[,2:227]


############################################################
############## Imputation using default ##############
############################################################

## Impute missing values
 imp <- mice(data_miss1, seed = 23109)
# 
imp_data1 <- complete(imp,1)
imp_data2 <- complete(imp,2)
imp_data3 <- complete(imp,3)
imp_data4 <- complete(imp,4)
imp_data5 <- complete(imp,5)

write.csv(imp_data1, file = "imp_data1_nonstand.csv")
write.csv(imp_data2, file = "imp_data2_nonstand.csv")
write.csv(imp_data3, file = "imp_data3_nonstand.csv")
write.csv(imp_data4, file = "imp_data4_nonstand.csv")
write.csv(imp_data5, file = "imp_data5_nonstand.csv")

## Load imputed data files

imp_data1 <- read.csv('imp_data1_nonstand.csv')
imp_data2 <- read.csv('imp_data2_nonstand.csv')
imp_data3 <- read.csv('imp_data3_nonstand.csv')
imp_data4 <- read.csv('imp_data4_nonstand.csv')
imp_data5 <- read.csv('imp_data5_nonstand.csv')

# Check imputed values are realistic and in line with previous distribution

# - check mean, variance and range of columns before and after imputation
# - check histograms

data_mean = colMeans(data_miss1,na.rm=TRUE)

imp1_mean = colMeans(imp_data1[,2:227],na.rm=FALSE)
imp2_mean = colMeans(imp_data2[,2:227],na.rm=FALSE)
imp3_mean = colMeans(imp_data3[,2:227],na.rm=FALSE)
imp4_mean = colMeans(imp_data4[,2:227],na.rm=FALSE)
imp5_mean = colMeans(imp_data5[,2:227],na.rm=FALSE)

data_mean_s = data_mean[data_mean<2]
imp1_mean_s = imp1_mean[data_mean<2]
imp2_mean_s = imp2_mean[data_mean<2]
imp3_mean_s = imp3_mean[data_mean<2]
imp4_mean_s = imp4_mean[data_mean<2]
imp5_mean_s = imp5_mean[data_mean<2]

data_mean_l = data_mean[data_mean>=2]
imp1_mean_l = imp1_mean[data_mean>=2]
imp2_mean_l = imp2_mean[data_mean>=2]
imp3_mean_l = imp3_mean[data_mean>=2]
imp4_mean_l = imp4_mean[data_mean>=2]
imp5_mean_l = imp5_mean[data_mean>=2]

plot(data_mean_s,imp1_mean_s)
plot(data_mean_s,imp2_mean_s)
plot(data_mean_s,imp3_mean_s)
plot(data_mean_s,imp4_mean_s)
plot(data_mean_s,imp5_mean_s)

plot(data_mean_l,imp1_mean_l)
plot(data_mean_l,imp2_mean_l)
plot(data_mean_l,imp3_mean_l)
plot(data_mean_l,imp4_mean_l)
plot(data_mean_l,imp5_mean_l)

# Order variables by missingness
data_miss1
missing_sum = sort(colSums(is.na(data_miss1)),decreasing = TRUE)

#Plot histograms to check distribution of most highly imputed columns (>100 missing, next column only 55 missing)
# plots rea all fine, some variability but not too bad :)
par(mfrow=c(2,3))
hist(data_miss1[,'L.HDL.PL_.'])
hist(imp_data1[,'L.HDL.PL_.'])
hist(imp_data2[,'L.HDL.PL_.'])
hist(imp_data3[,'L.HDL.PL_.'])
hist(imp_data4[,'L.HDL.PL_.'])
hist(imp_data5[,'L.HDL.PL_.'])

hist(data_miss1[,'L.HDL.C_.'])
hist(imp_data1[,'L.HDL.C_.'])
hist(imp_data2[,'L.HDL.C_.'])
hist(imp_data3[,'L.HDL.C_.'])
hist(imp_data4[,'L.HDL.C_.'])
hist(imp_data5[,'L.HDL.C_.'])
#large amounts of low values in 3,4,5?

hist(data_miss1[,'L.HDL.CE_.'])
hist(imp_data1[,'L.HDL.CE_.'])
hist(imp_data2[,'L.HDL.CE_.'])
hist(imp_data3[,'L.HDL.CE_.'])
hist(imp_data4[,'L.HDL.CE_.'])
hist(imp_data5[,'L.HDL.CE_.'])
# same again

hist(data_miss1[,'L.HDL.FC_.'])
hist(imp_data1[,'L.HDL.FC_.'])
hist(imp_data2[,'L.HDL.FC_.'])
hist(imp_data3[,'L.HDL.FC_.'])
hist(imp_data4[,'L.HDL.FC_.'])
hist(imp_data5[,'L.HDL.FC_.'])
# again 4,5

hist(data_miss1[,'L.HDL.TG_.'])
hist(imp_data1[,'L.HDL.TG_.'])
hist(imp_data2[,'L.HDL.TG_.'])
hist(imp_data3[,'L.HDL.TG_.'])
hist(imp_data4[,'L.HDL.TG_.'])
hist(imp_data5[,'L.HDL.TG_.'])

hist(data_miss1[,'XL.VLDL.PL_.'])
hist(imp_data1[,'XL.VLDL.PL_.'])
hist(imp_data2[,'XL.VLDL.PL_.'])
hist(imp_data3[,'XL.VLDL.PL_.'])
hist(imp_data4[,'XL.VLDL.PL_.'])
hist(imp_data5[,'XL.VLDL.PL_.'])

hist(data_miss1[,'XL.VLDL.C_.'])
hist(imp_data1[,'XL.VLDL.C_.'])
hist(imp_data2[,'XL.VLDL.C_.'])
hist(imp_data3[,'XL.VLDL.C_.'])
hist(imp_data4[,'XL.VLDL.C_.'])
hist(imp_data5[,'XL.VLDL.C_.'])

hist(data_miss1[,'XL.VLDL.FC_.'])
hist(imp_data1[,'XL.VLDL.FC_.'])
hist(imp_data2[,'XL.VLDL.FC_.'])
hist(imp_data3[,'XL.VLDL.FC_.'])
hist(imp_data4[,'XL.VLDL.FC_.'])
hist(imp_data5[,'XL.VLDL.FC_.'])

hist(data_miss1[,'XL.VLDL.TG_.'])
hist(imp_data1[,'XL.VLDL.TG_.'])
hist(imp_data2[,'XL.VLDL.TG_.'])
hist(imp_data3[,'XL.VLDL.TG_.'])
hist(imp_data4[,'XL.VLDL.TG_.'])
hist(imp_data5[,'XL.VLDL.TG_.'])

hist(data_miss1[,'Gln'])
hist(imp_data1[,'Gln'])
hist(imp_data2[,'Gln'])
hist(imp_data3[,'Gln'])
hist(imp_data4[,'Gln'])
hist(imp_data5[,'Gln'])

hist(data_miss1[,'XL.HDL.PL_.'])
hist(imp_data1[,'XL.HDL.PL_.'])
hist(imp_data2[,'XL.HDL.PL_.'])
hist(imp_data3[,'XL.HDL.PL_.'])
hist(imp_data4[,'XL.HDL.PL_.'])
hist(imp_data5[,'XL.HDL.PL_.'])

hist(data_miss1[,'XL.HDL.C_.'])
hist(imp_data1[,'XL.HDL.C_.'])
hist(imp_data2[,'XL.HDL.C_.'])
hist(imp_data3[,'XL.HDL.C_.'])
hist(imp_data4[,'XL.HDL.C_.'])
hist(imp_data5[,'XL.HDL.C_.'])

hist(data_miss1[,'XL.HDL.CE_.'])
hist(imp_data1[,'XL.HDL.CE_.'])
hist(imp_data2[,'XL.HDL.CE_.'])
hist(imp_data3[,'XL.HDL.CE_.'])
hist(imp_data4[,'XL.HDL.CE_.'])
hist(imp_data5[,'XL.HDL.CE_.'])

hist(data_miss1[,'XL.HDL.FC_.'])
hist(imp_data1[,'XL.HDL.FC_.'])
hist(imp_data2[,'XL.HDL.FC_.'])
hist(imp_data3[,'XL.HDL.FC_.'])
hist(imp_data4[,'XL.HDL.FC_.'])
hist(imp_data5[,'XL.HDL.FC_.'])

hist(data_miss1[,'XL.HDL.TG_.'])
hist(imp_data1[,'XL.HDL.TG_.'])
hist(imp_data2[,'XL.HDL.TG_.'])
hist(imp_data3[,'XL.HDL.TG_.'])
hist(imp_data4[,'XL.HDL.TG_.'])
hist(imp_data5[,'XL.HDL.TG_.'])

hist(data_miss1[,'XXL.VLDL.PL_.'])
hist(imp_data1[,'XXL.VLDL.PL_.'])
hist(imp_data2[,'XXL.VLDL.PL_.'])
hist(imp_data3[,'XXL.VLDL.PL_.'])
hist(imp_data4[,'XXL.VLDL.PL_.'])
hist(imp_data5[,'XXL.VLDL.PL_.'])

hist(data_miss1[,'XXL.VLDL.C_.'])
hist(imp_data1[,'XXL.VLDL.C_.'])
hist(imp_data2[,'XXL.VLDL.C_.'])
hist(imp_data3[,'XXL.VLDL.C_.'])
hist(imp_data4[,'XXL.VLDL.C_.'])
hist(imp_data5[,'XXL.VLDL.C_.'])

hist(data_miss1[,'XXL.VLDL.CE_.'])
hist(imp_data1[,'XXL.VLDL.CE_.'])
hist(imp_data2[,'XXL.VLDL.CE_.'])
hist(imp_data3[,'XXL.VLDL.CE_.'])
hist(imp_data4[,'XXL.VLDL.CE_.'])
hist(imp_data5[,'XXL.VLDL.CE_.'])

hist(data_miss1[,'XXL.VLDL.FC_.'])
hist(imp_data1[,'XXL.VLDL.FC_.'])
hist(imp_data2[,'XXL.VLDL.FC_.'])
hist(imp_data3[,'XXL.VLDL.FC_.'])
hist(imp_data4[,'XXL.VLDL.FC_.'])
hist(imp_data5[,'XXL.VLDL.FC_.'])

hist(data_miss1[,'XXL.VLDL.TG_.'])
hist(imp_data1[,'XXL.VLDL.TG_.'])
hist(imp_data2[,'XXL.VLDL.TG_.'])
hist(imp_data3[,'XXL.VLDL.TG_.'])
hist(imp_data4[,'XXL.VLDL.TG_.'])
hist(imp_data5[,'XXL.VLDL.TG_.'])

hist(data_miss1[,'bOHBut'])
hist(imp_data1[,'bOHBut'])
hist(imp_data2[,'bOHBut'])
hist(imp_data3[,'bOHBut'])
hist(imp_data4[,'bOHBut'])
hist(imp_data5[,'bOHBut'])

hist(data_miss1[,'L.VLDL.PL_.'])
hist(imp_data1[,'L.VLDL.PL_.'])
hist(imp_data2[,'L.VLDL.PL_.'])
hist(imp_data3[,'L.VLDL.PL_.'])
hist(imp_data4[,'L.VLDL.PL_.'])
hist(imp_data5[,'L.VLDL.PL_.'])

hist(data_miss1[,'L.VLDL.C_.'])
hist(imp_data1[,'L.VLDL.C_.'])
hist(imp_data2[,'L.VLDL.C_.'])
hist(imp_data3[,'L.VLDL.C_.'])
hist(imp_data4[,'L.VLDL.C_.'])
hist(imp_data5[,'L.VLDL.C_.'])

hist(data_miss1[,'L.VLDL.CE_.'])
hist(imp_data1[,'L.VLDL.CE_.'])
hist(imp_data2[,'L.VLDL.CE_.'])
hist(imp_data3[,'L.VLDL.CE_.'])
hist(imp_data4[,'L.VLDL.CE_.'])
hist(imp_data5[,'L.VLDL.CE_.'])

hist(data_miss1[,'L.VLDL.FC_.'])
hist(imp_data1[,'L.VLDL.FC_.'])
hist(imp_data2[,'L.VLDL.FC_.'])
hist(imp_data3[,'L.VLDL.FC_.'])
hist(imp_data4[,'L.VLDL.FC_.'])
hist(imp_data5[,'L.VLDL.FC_.'])
# large amount of low values 4,5

hist(data_miss1[,'L.VLDL.TG_.'])
hist(imp_data1[,'L.VLDL.TG_.'])
hist(imp_data2[,'L.VLDL.TG_.'])
hist(imp_data3[,'L.VLDL.TG_.'])
hist(imp_data4[,'L.VLDL.TG_.'])
hist(imp_data5[,'L.VLDL.TG_.'])




