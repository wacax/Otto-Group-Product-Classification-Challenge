#Otto Group Product Classification Challenge
#Ver 0.3.0 # h2o deep nets improved + PCA and feature selection

#Init-----------------------------------------------
rm(list=ls(all=TRUE))

#Libraries, directories, options and extra functions----------------------
require("data.table")
require("h2o")
require("leaps")

#Set Working Directory
workingDirectory <- "/home/wacax/Wacax/Kaggle/Otto/"
setwd(workingDirectory)
dataDirectory <- "/home/wacax/Wacax/Kaggle/Otto/Data/"
#h2o location
h2o.jarLoc <- "/home/wacax/R/x86_64-pc-linux-gnu-library/3.1/h2o/java/h2o.jar"

#Load data
#train <- fread(file.path(dataDirectory, "train.csv"), verbose = TRUE)
#test <- fread(file.path(dataDirectory, "test.csv"), verbose = TRUE)

#EDA----------------------------------
## EDA Pt. 1 Determine the minimal PCAs / number of neurons in the middle layer
#Start h2o from command line
system(paste0("java -Xmx5G -jar ", h2o.jarLoc, " -port 54333 -name Otto &"))
#Small pause
Sys.sleep(3)
#Connect R to h2o
h2oServer <- h2o.init(ip = "localhost", port = 54333, nthreads = -1)

#R data.table to h2o.ai
h2oOttoTrain <- h2o.importFile(h2oServer, file.path(dataDirectory, "train.csv"))

#PCA
PCAModel <- h2o.prcomp(h2oOttoTrain)
plot(PCAModel@model$sdev)
#ggplot(data.frame(X = PCAModel@model$sdev), aes(x = X)) + geom_density()
h2o.shutdown(h2oServer, prompt = FALSE)

## EDA Pt. 2 Select the best features
#Load data
train <- as.data.frame(fread(file.path(dataDirectory, "train.csv"), verbose = TRUE))
#Shuffle indexes
#set.seed(1001001)
andIdxs <- sample(seq(1, nrow(h2oOttoTrain)), nrow(h2oOttoTrain))

linearBestModels <- regsubsets(x = as.matrix(train[randIdxs, seq(2, ncol(train) - 1)]),
                               y = as.factor(train[randIdxs, ncol(train)]), 
                               method = "forward", nvmax=90)

#Plot the best number of predictors
bestMods <- summary(linearBestModels)
bestNumberOfPredictors <- which.min(bestMods$cp)
plot(bestMods$cp, xlab="Number of Variables", ylab="CP Error", main ="Best Number of Features")
points(bestNumberOfPredictors, bestMods$cp[bestNumberOfPredictors], pch=20, col="red")

#Name of the most predictive rankings
predictors1 <- as.data.frame(bestMods$which)
bestFeatures <- names(sort(apply(predictors1[, -1], 2, sum), decreasing = TRUE)[1:bestNumberOfPredictors])

#RF Modelling--------------------------
#Start h2o from command line
system(paste0("java -Xmx5G -jar ", h2o.jarLoc, " -port 54333 -name Otto &"))
#Small pause
Sys.sleep(3)
#Connect R to h2o
h2oServer <- h2o.init(ip = "localhost", port = 54333, nthreads = -1)

#R data.table to h2o.ai
h2oOttoTrain <- h2o.importFile(h2oServer, file.path(dataDirectory, "train.csv"))
h2oOttoTest <- h2o.importFile(h2oServer, file.path(dataDirectory, "test.csv"))

#Shuffle indexes
#set.seed(1001001)
randIdxs <- sample(seq(1, nrow(h2oOttoTrain)), nrow(h2oOttoTrain))
#Cross Validation
ottoRFModelCV <- h2o.randomForest(x = seq(2, ncol(h2oOttoTrain)), y = ncol(h2oOttoTrain),
                                  data = h2oOttoTrain[randIdxs, ],
                                  nfolds = 5,
                                  classification = TRUE,
                                  ntree = c(50, 75, 100),
                                  depth = c(20, 50), 
                                  verbose = FALSE)

#Log Info
mseRF <- ottoRFModelCV@model[[1]]@model$mse
ntreeRF <- ottoRFModelCV@model[[1]]@model$params$ntree
depthRF <- ottoRFModelCV@model[[1]]@model$params$depth

#h2o.ai RF Modelling    
ottoRFModel <- h2o.randomForest(x = seq(2, ncol(h2oOttoTrain)), y = ncol(h2oOttoTrain),
                                data = h2oOttoTrain[randIdxs, ],
                                classification = TRUE,
                                ntree = ntreeRF,
                                depth = depthRF,
                                type = "BigData",
                                verbose = FALSE)

#probability Prediction of trips in Nth driver 
predictionRF <- as.data.frame(h2o.predict(ottoRFModel, newdata = h2oOttoTest)[, seq(2, 10)])
h2o.rm(object = h2oServer, keys = h2o.ls(h2oServer)[, 1])   
print(paste0("Data processed with RFs")) 

#Shutdown h20 instance
h2o.shutdown(h2oServer, prompt = FALSE)

#Make a submission file
sampleSubmissionFile <- fread(file.path(dataDirectory, "sampleSubmission.csv"), verbose = TRUE)
sampleSubmissionFile <- as.data.frame(sampleSubmissionFile)
sampleSubmissionFile[, seq(2, 10)] <- predictionRF

write.csv(sampleSubmissionFile, file = "RFPredictionII.csv", row.names = FALSE)
system('zip RFPredictionII.zip RFPredictionII.csv')

#GBM Modeling---------------------
#Start h2o from command line
system(paste0("java -Xmx5G -jar ", h2o.jarLoc, " -port 54333 -name Otto &"))
#Small pause
Sys.sleep(3)
#Connect R to h2o
h2oServer <- h2o.init(ip = "localhost", port = 54333, nthreads = -1)

#R data.table to h2o.ai
h2oOttoTrain <- h2o.importFile(h2oServer, file.path(dataDirectory, "train.csv"))
h2oOttoTest <- h2o.importFile(h2oServer, file.path(dataDirectory, "test.csv"))

#Shuffle indexes
#set.seed(1001001)
randIdxs <- sample(seq(1, nrow(h2oOttoTrain)), nrow(h2oOttoTrain))
#Cross Validation
ottoGBMModelCV <- h2o.gbm(x = seq(2, ncol(h2oOttoTrain)), y = ncol(h2oOttoTrain),
                          data = h2oOttoTrain[randIdxs, ],
                          nfolds = 5,
                          distribution = "multinomial",
                          interaction.depth = c(2, 5, 7),
                          shrinkage = c(0.001, 0.003), 
                          n.trees = 500,                           
                          grid.parallelism = 4)

#Log Info
mseGBM <- mean(ottoGBMModelCV@model[[1]]@model$err)
interaction.depthGBM <- ottoGBMModelCV@model[[1]]@model$params$interaction
shrinkageGBM <- ottoGBMModelCV@model[[1]]@model$params$shrinkage

#h2o.ai GBM Modelling    
ottoGBMModel <- h2o.gbm(x = seq(2, ncol(h2oOttoTrain)), y = ncol(h2oOttoTrain),
                        data = h2oOttoTrain[randIdxs, ],
                        distribution = "multinomial",
                        interaction.depth = interaction.depthGBM,
                        shrinkage = shrinkageGBM, 
                        n.trees = 4000)

#probability Prediction of trips in Nth driver 
predictionGBM <- as.data.frame(h2o.predict(ottoGBMModel, newdata = h2oOttoTest)[, seq(2, 10)])
h2o.rm(object = h2oServer, keys = h2o.ls(h2oServer)[, 1])   
print(paste0("Data processed with GBM")) 

#Shutdown h20 instance
h2o.shutdown(h2oServer, prompt = FALSE)

#Make a submission file
sampleSubmissionFile <- fread(file.path(dataDirectory, "sampleSubmission.csv"), verbose = TRUE)
sampleSubmissionFile <- as.data.frame(sampleSubmissionFile)
sampleSubmissionFile[, seq(2, 10)] <- predictionGBM

write.csv(sampleSubmissionFile, file = "GBMPredictionII.csv", row.names = FALSE)
system('zip GBMPredictionII.zip GBMPredictionII.csv')


#Deep Nets Modeling---------------------
#Start h2o from command line
system(paste0("java -Xmx5G -jar ", h2o.jarLoc, " -port 54333 -name Otto &"))
#Small pause
Sys.sleep(3)
#Connect R to h2o
h2oServer <- h2o.init(ip = "localhost", port = 54333, nthreads = -1)

#R data.table to h2o.ai
h2oOttoTrain <- h2o.importFile(h2oServer, file.path(dataDirectory, "train.csv"))
h2oOttoTest <- h2o.importFile(h2oServer, file.path(dataDirectory, "test.csv"))

#Shuffle indexes
#set.seed(1001001)
randIdxs <- sample(seq(1, nrow(h2oOttoTrain)), nrow(h2oOttoTrain))
#Cross Validation
ottoDeepNetModelCV <- h2o.deeplearning(x = seq(2, ncol(h2oOttoTrain)), y = ncol(h2oOttoTrain),
                                       data = h2oOttoTrain[randIdxs, ],
                                       nfolds = 5,
                                       classification = TRUE,
                                       activation = c("TanhWithDropout", "RectifierWithDropout"),
                                       hidden = c(105, 105, 105), 
                                       hidden_dropout_ratios = list(c(0, 0, 0), c(0.5, 0.5, 0.5)),
                                       adaptive_rate = TRUE, 
                                       rho = c(0.99, 0.95), 
                                       epsilon = c(1e-10, 1e-8, 1e-6), 
                                       nesterov_accelerated_gradient = TRUE)

#Log Info
errorDeepNN <- ottoDeepNetModelCV@model[[1]]@model$valid_class_error
bestActivationNN <- ottoDeepNetModelCV@model[[1]]@model$params$activation
bestHdr <- ottoDeepNetModelCV@model[[1]]@model$params$hidden_dropout_ratios
bestRho <- ottoDeepNetModelCV@model[[1]]@model$params$rho
bestEpsilon <- ottoDeepNetModelCV@model[[1]]@model$params$epsilon

#h2o.ai deep NN Modelling    
ottoDeepNetModel <- h2o.deeplearning(x = seq(2, ncol(h2oOttoTrain)), y = ncol(h2oOttoTrain),
                                     data = h2oOttoTrain[randIdxs, ],
                                     classification = TRUE,
                                     activation = bestActivationNN,
                                     hidden = c(105, 105, 105),
                                     hidden_dropout_ratios = bestHdr, 
                                     adaptive_rate = TRUE, 
                                     rho = bestRho, 
                                     epsilon = bestEpsilon, 
                                     nesterov_accelerated_gradient = TRUE, 
                                     epochs = 200)

#probability Prediction nth category
predictionNN <- as.data.frame(h2o.predict(ottoDeepNetModel, newdata = h2oOttoTest)[, seq(2, 10)])
h2o.rm(object = h2oServer, keys = h2o.ls(h2oServer)[, 1]) 
print(paste0("Data processed with NNs")) 

#Shutdown h20 instance
h2o.shutdown(h2oServer, prompt = FALSE)

#Make submission files
sampleSubmissionFile <- fread(file.path(dataDirectory, "sampleSubmission.csv"), verbose = TRUE)
sampleSubmissionFile <- as.data.frame(sampleSubmissionFile)
sampleSubmissionFile[, seq(2, 10)] <- predictionNN

write.csv(sampleSubmissionFile, file = "NNPredictionII.csv", row.names = FALSE)
system('zip NNPredictionII.zip NNPredictionII.csv')

#Deep Nets with feature selection
#Start h2o from command line
system(paste0("java -Xmx5G -jar ", h2o.jarLoc, " -port 54333 -name Otto &"))
#Small pause
Sys.sleep(3)
#Connect R to h2o
h2oServer <- h2o.init(ip = "localhost", port = 54333, nthreads = -1)

#R data.table to h2o.ai
h2oOttoTrain <- h2o.importFile(h2oServer, file.path(dataDirectory, "train.csv"))
h2oOttoTest <- h2o.importFile(h2oServer, file.path(dataDirectory, "test.csv"))

#Shuffle indexes
#set.seed(1001001)
randIdxs <- sample(seq(1, nrow(h2oOttoTrain)), nrow(h2oOttoTrain))

#Cross Validation, model with best features
bestFeaturesIdx <- which(names(h2oOttoTrain) %in% bestFeatures)
ottoDeepNetModelBestFeatCV <- h2o.deeplearning(x = bestFeaturesIdx, y = ncol(h2oOttoTrain),
                                               data = h2oOttoTrain[randIdxs, ],
                                               nfolds = 5,
                                               classification = TRUE,
                                               activation = c("TanhWithDropout", "RectifierWithDropout"),
                                               hidden = cc(105, 105, 105), 
                                               hidden_dropout_ratios = list(c(0, 0, 0), c(0.5, 0.5, 0.5)),
                                               adaptive_rate = TRUE, 
                                               rho = c(0.99, 0.95), 
                                               epsilon = c(1e-10, 1e-8, 1e-6), 
                                               nesterov_accelerated_gradient = TRUE)

#Log info
errorDeepNNBestFeat <- ottoDeepNetModelBestFeatCV@model[[1]]@model$valid_class_error
bestActivationNNBestFeat <- ottoDeepNetModelBestFeatCV@model[[1]]@model$params$activation
bestHdrBestFeat <- ottoDeepNetModelBestFeatCV@model[[1]]@model$params$hidden_dropout_ratios
bestRhoBestFeat <- ottoDeepNetModelBestFeatCV@model[[1]]@model$params$rho
bestEpsilonBestFeat <- ottoDeepNetModelBestFeatCV@model[[1]]@model$params$epsilon

ottoDeepNetModelBestFeat <- h2o.deeplearning(x = bestFeaturesIdx, y = ncol(h2oOttoTrain),
                                             data = h2oOttoTrain[randIdxs, ],
                                             classification = TRUE,
                                             activation = bestActivationNNBestFeat,
                                             hidden = c(105, 105, 105),
                                             hidden_dropout_ratios = bestHdrBestFeat, 
                                             adaptive_rate = TRUE, 
                                             rho = bestRhoBestFeat, 
                                             epsilon = bestEpsilonBestFeat, 
                                             nesterov_accelerated_gradient = TRUE, 
                                             epochs = 200)

#probability Prediction nth category best features
predictionNNBestFeat  <- as.data.frame(h2o.predict(ottoDeepNetModelBestFeat, newdata = h2oOttoTest)[, seq(2, 10)])
h2o.rm(object = h2oServer, keys = h2o.ls(h2oServer)[, 1]) 
print(paste0("Data processed with NNs")) 

#Shutdown h20 instance
h2o.shutdown(h2oServer, prompt = FALSE)

sampleSubmissionFile <- fread(file.path(dataDirectory, "sampleSubmission.csv"), verbose = TRUE)
sampleSubmissionFile <- as.data.frame(sampleSubmissionFile)
sampleSubmissionFile[, seq(2, 10)] <- predictionNNBestFeat

write.csv(sampleSubmissionFile, file = "NNPredictionBestFeatsII.csv", row.names = FALSE)
system("zip NNPredictionBestFeatsII.zip NNPredictionBestFeatsII.csv")

