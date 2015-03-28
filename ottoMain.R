#Otto Group Product Classification Challenge
#Ver 0.4.0 # ensemble

#Init-----------------------------------------------
rm(list=ls(all=TRUE))

#Libraries, directories, options and extra functions----------------------
require("data.table")
require("h2o")
require("leaps")
require("Metrics")

#Set Working Directory
workingDirectory <- "/home/wacax/Wacax/Kaggle/Otto/"
setwd(workingDirectory)
dataDirectory <- "/home/wacax/Wacax/Kaggle/Otto/Data/"
#h2o location
h2o.jarLoc <- "/home/wacax/R/x86_64-pc-linux-gnu-library/3.1/h2o/java/h2o.jar"

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
randIdxs <- sample(seq(1, nrow(train)), nrow(train))

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
rm(train)

#Data Split----------------------------
#Load data
train <- fread(file.path(dataDirectory, "train.csv"), verbose = TRUE)
#Length Data Splits
cvTrainLength <- floor(floor(nrow(train) * 0.6) * 0.15)   #15% of the 60% modeling data
cvValidationLength <- floor(floor(nrow(train) * 0.6) * 0.85)   #85% of the 60% modeling data
ensembleCvLength <- floor(floor(nrow(train) * 0.4) * 0.2)   #20% of the 40% ensemble data
ensembleValidationLength <- floor(floor(nrow(train) * 0.4) * 0.8)   #80% of the 40% ensemble9 data

idxsdiff <- nrow(train) - length(c(rep(1, cvTrainLength), rep(2, cvValidationLength), 
                                   rep(3, ensembleCvLength), rep(4, ensembleValidationLength))) 
groupsVector <- sample(c(rep(1, cvTrainLength), rep(2, cvValidationLength), 
                         rep(3, ensembleCvLength), rep(4, ensembleValidationLength + idxsdiff)), 
                       nrow(train))

#Shuffle Indices
set.seed(1001001)
dataSplits <- split(seq(1, nrow(train)), as.factor(groupsVector))
rm(train)

#RF Modelling--------------------------
#Start h2o from command line
system(paste0("java -Xmx5G -jar ", h2o.jarLoc, " -port 54333 -name Otto &"))
#Small pause
Sys.sleep(3)
#Connect R to h2o
h2oServer <- h2o.init(ip = "localhost", port = 54333, nthreads = -1)

#R data.table to h2o.ai
h2oOttoTrain <- h2o.importFile(h2oServer, file.path(dataDirectory, "train.csv"))

#Shuffle indexes
#Cross Validation
ottoRFModelCV <- h2o.randomForest(x = seq(2, ncol(h2oOttoTrain)), y = ncol(h2oOttoTrain),
                                  data = h2oOttoTrain[dataSplits[[1]], ],
                                  nfolds = 5,
                                  classification = TRUE,
                                  ntree = c(75, 100, 150),
                                  depth = c(50, 75), 
                                  verbose = FALSE)

#Log Info
mseRF <- ottoRFModelCV@model[[1]]@model$mse
ntreeRF <- ottoRFModelCV@model[[1]]@model$params$ntree
depthRF <- ottoRFModelCV@model[[1]]@model$params$depth
print(paste0("RFs best parameters found")) 

#h2o.ai RF Modelling    
ottoRFModel <- h2o.randomForest(x = seq(2, ncol(h2oOttoTrain)), y = ncol(h2oOttoTrain),
                                data = h2oOttoTrain[c(dataSplits[[1]], dataSplits[[2]]), ],
                                classification = TRUE,
                                ntree = ntreeRF,
                                depth = depthRF,
                                type = "BigData",
                                verbose = FALSE)

#probability Prediction 
predictionRFValidation <- as.data.frame(h2o.predict(ottoRFModel, newdata = h2oOttoTrain[c(dataSplits[[3]], dataSplits[[4]]), ])[, seq(2, 10)])
print(paste0("RFs model built")) 

#Save model
h2o.saveModel(ottoRFModel, dir = dataDirectory, name = "ModelRF", save_cv = FALSE, force = FALSE)
h2o.rm(object = h2oServer, keys = h2o.ls(h2oServer)[, 1])   

#Shutdown h20 instance
h2o.shutdown(h2oServer, prompt = FALSE)

#Make a submission file
write.csv(predictionRFValidation, file = "RFValidation.csv", row.names = FALSE)

#GBM Modeling---------------------
#Start h2o from command line
system(paste0("java -Xmx5G -jar ", h2o.jarLoc, " -port 54333 -name Otto &"))
#Small pause
Sys.sleep(3)
#Connect R to h2o
h2oServer <- h2o.init(ip = "localhost", port = 54333, nthreads = -1)

#R data.table to h2o.ai
h2oOttoTrain <- h2o.importFile(h2oServer, file.path(dataDirectory, "train.csv"))

#Cross Validation
ottoGBMModelCV <- h2o.gbm(x = seq(2, ncol(h2oOttoTrain)), y = ncol(h2oOttoTrain),
                          data = h2oOttoTrain[dataSplits[[1]], ],
                          nfolds = 5,
                          distribution = "multinomial",
                          interaction.depth = c(6, 8, 10),
                          shrinkage = c(0.001, 0.003), 
                          n.trees = 500,                           
                          grid.parallelism = 4)

#Log Info
mseGBM <- mean(ottoGBMModelCV@model[[1]]@model$err)
interaction.depthGBM <- ottoGBMModelCV@model[[1]]@model$params$interaction
shrinkageGBM <- ottoGBMModelCV@model[[1]]@model$params$shrinkage

#h2o.ai GBM Modelling    
ottoGBMModel <- h2o.gbm(x = seq(2, ncol(h2oOttoTrain)), y = ncol(h2oOttoTrain),
                        data = h2oOttoTrain[c(dataSplits[[1]], dataSplits[[2]]), ],
                        distribution = "multinomial",
                        interaction.depth = interaction.depthGBM,
                        shrinkage = shrinkageGBM, 
                        n.trees = 8000)

#probability Prediction of trips in Nth driver 
predictionGBMValidation <- as.data.frame(h2o.predict(ottoGBMModel, newdata = h2oOttoTrain[c(dataSplits[[3]], dataSplits[[4]]), ])[, seq(2, 10)])
print(paste0("Data processed with GBM")) 

#Save model
h2o.saveModel(ottoGBMModel, dir = dataDirectory, name = "ModelGBM", save_cv = FALSE, force = FALSE)
h2o.rm(object = h2oServer, keys = h2o.ls(h2oServer)[, 1])   

#Shutdown h20 instance
h2o.shutdown(h2oServer, prompt = FALSE)

#Make a submission file
write.csv(predictionGBMValidation, file = "GBMValidation.csv", row.names = FALSE)

#Deep Nets Modeling---------------------
#Start h2o from command line
system(paste0("java -Xmx5G -jar ", h2o.jarLoc, " -port 54333 -name Otto &"))
#Small pause
Sys.sleep(3)
#Connect R to h2o
h2oServer <- h2o.init(ip = "localhost", port = 54333, nthreads = -1)

#R data.table to h2o.ai
#Load data
train <- as.data.frame(fread(file.path(dataDirectory, "train.csv"), verbose = TRUE))
test <- as.data.frame(fread(file.path(dataDirectory, "test.csv"), verbose = TRUE))

scaledFullData <- signif(scale(rbind(train[, seq(2, ncol(train) - 1)], test[, seq(2, ncol(test))])), digits = 5)

train[, seq(2, ncol(train) - 1)] <- scaledFullData[seq(1, nrow(train)), ]
test[, seq(2, ncol(test))] <- scaledFullData[seq(nrow(train) + 1, nrow(scaledFullData)), ]

h2oOttoTrain <- as.h2o(h2oServer, train)

rm(train, test, scaledFullData)

#Cross Validation
ottoDeepNetModelCV <- h2o.deeplearning(x = seq(2, ncol(h2oOttoTrain)), y = ncol(h2oOttoTrain),
                                       data = h2oOttoTrain[dataSplits[[1]], ],
                                       nfolds = 5,
                                       classification = TRUE,
                                       activation = c("Rectifier", "Tanh"),
                                       hidden = c(100, 100, 100), 
                                       hidden_dropout_ratios = list(c(0, 0, 0), c(0.5, 0.5, 0.5)),
                                       adaptive_rate = TRUE, 
                                       rho = c(0.99, 0.95), 
                                       epsilon = c(1e-10, 1e-8, 1e-6), 
                                       l1 = c(0, 1e-6),
                                       l2 = c(0, 1e-6),                                       
                                       nesterov_accelerated_gradient = TRUE, 
                                       epochs = 10)

#Log Info
errorDeepNN <- ottoDeepNetModelCV@model[[1]]@model$valid_class_error
bestActivationNN <- ottoDeepNetModelCV@model[[1]]@model$params$activation
bestHdr <- ottoDeepNetModelCV@model[[1]]@model$params$hidden_dropout_ratios
bestRho <- ottoDeepNetModelCV@model[[1]]@model$params$rho
bestEpsilon <- ottoDeepNetModelCV@model[[1]]@model$params$epsilon
bestl1 <- ottoDeepNetModelCV@model[[1]]@model$params$l1
bestl2 <- ottoDeepNetModelCV@model[[1]]@model$params$l2

#h2o.ai deep NN Modelling    
ottoDeepNetModel <- h2o.deeplearning(x = seq(2, ncol(h2oOttoTrain)), y = ncol(h2oOttoTrain),
                                     data = h2oOttoTrain[c(dataSplits[[1]], dataSplits[[2]]), ],
                                     classification = TRUE,
                                     activation = bestActivationNN,
                                     hidden = c(100, 100, 100),
                                     hidden_dropout_ratios = bestHdr, 
                                     adaptive_rate = TRUE, 
                                     rho = bestRho, 
                                     epsilon = bestEpsilon, 
                                     nesterov_accelerated_gradient = TRUE, 
                                     l1 = bestl1, 
                                     l2 = bestl2,
                                     epochs = 400)

#probability Prediction nth category
predictionNNValidation <- as.data.frame(h2o.predict(ottoDeepNetModel, newdata = h2oOttoTrain[c(dataSplits[[3]], dataSplits[[4]]), ])[, seq(2, 10)])
print(paste0("Data processed with NNs")) 

#Save model
h2o.saveModel(ottoDeepNetModel, dir = dataDirectory, name = "ModelNN", save_cv = FALSE, force = FALSE)
h2o.rm(object = h2oServer, keys = h2o.ls(h2oServer)[, 1])   

#Shutdown h20 instance
h2o.shutdown(h2oServer, prompt = FALSE)

#Make a submission file
write.csv(predictionNNValidation, file = "NNValidation.csv", row.names = FALSE)

#Deep Nets with feature selection
#Start h2o from command line
system(paste0("java -Xmx5G -jar ", h2o.jarLoc, " -port 54333 -name Otto &"))
#Small pause
Sys.sleep(3)
#Connect R to h2o
h2oServer <- h2o.init(ip = "localhost", port = 54333, nthreads = -1)

#R data.table to h2o.ai
#Load data
train <- as.data.frame(fread(file.path(dataDirectory, "train.csv"), verbose = TRUE))
test <- as.data.frame(fread(file.path(dataDirectory, "test.csv"), verbose = TRUE))

scaledFullData <- signif(scale(rbind(train[, seq(2, ncol(train) - 1)], test[, seq(2, ncol(test))])), digits = 5)

train[, seq(2, ncol(train) - 1)] <- scaledFullData[seq(1, nrow(train)), ]
test[, seq(2, ncol(test))] <- scaledFullData[seq(nrow(train) + 1, nrow(scaledFullData)), ]

h2oOttoTrain <- as.h2o(h2oServer, train)

rm(train, test, scaledFullData)

#Cross Validation, model with best features
bestFeaturesIdx <- which(names(h2oOttoTrain) %in% bestFeatures)
ottoDeepNetModelBestFeatCV <- h2o.deeplearning(x = bestFeaturesIdx, y = ncol(h2oOttoTrain),
                                               data = h2oOttoTrain[dataSplits[[3]], ],
                                               nfolds = 5,
                                               classification = TRUE,
                                               activation = c("Rectifier", "Tanh"),
                                               hidden = c(100, 100, 100), 
                                               hidden_dropout_ratios = list(c(0, 0, 0), c(0.5, 0.5, 0.5)),
                                               adaptive_rate = TRUE, 
                                               rho = c(0.99, 0.95), 
                                               epsilon = c(1e-10, 1e-8, 1e-6), 
                                               nesterov_accelerated_gradient = TRUE, 
                                               l1 = c(0, 1e-6),
                                               l2 = c(0, 1e-6),              
                                               epochs = 10)

#Log info
errorDeepNNBestFeat <- ottoDeepNetModelBestFeatCV@model[[1]]@model$valid_class_error
bestActivationNNBestFeat <- ottoDeepNetModelBestFeatCV@model[[1]]@model$params$activation
bestHdrBestFeat <- ottoDeepNetModelBestFeatCV@model[[1]]@model$params$hidden_dropout_ratios
bestRhoBestFeat <- ottoDeepNetModelBestFeatCV@model[[1]]@model$params$rho
bestEpsilonBestFeat <- ottoDeepNetModelBestFeatCV@model[[1]]@model$params$epsilon
bestl1BestFeat <- ottoDeepNetModelBestFeatCV@model[[1]]@model$params$l1
bestl2BestFeat <- ottoDeepNetModelBestFeatCV@model[[1]]@model$params$l2

ottoDeepNetModelBestFeat <- h2o.deeplearning(x = bestFeaturesIdx, y = ncol(h2oOttoTrain),
                                             data = h2oOttoTrain[c(dataSplits[[3]], dataSplits[[4]]), ],
                                             classification = TRUE,
                                             activation = bestActivationNNBestFeat,
                                             hidden = c(100, 100, 100), 
                                             hidden_dropout_ratios = bestHdrBestFeat, 
                                             adaptive_rate = TRUE, 
                                             rho = bestRhoBestFeat, 
                                             epsilon = bestEpsilonBestFeat, 
                                             nesterov_accelerated_gradient = TRUE, 
                                             l1 = bestl1BestFeat, 
                                             l2 = bestl2BestFeat,
                                             epochs = 400)

#probability Prediction nth category
predictionNNFeatValidation <- as.data.frame(h2o.predict(ottoDeepNetModelBestFeat, newdata = h2oOttoTrain[c(dataSplits[[3]], dataSplits[[4]]), ])[, seq(2, 10)])
print(paste0("Data processed with NNs and best features")) 

#Save model
h2o.saveModel(ottoDeepNetModelBestFeat, dir = dataDirectory, name = "ModelNNBestFeat", save_cv = FALSE, force = FALSE)
h2o.rm(object = h2oServer, keys = h2o.ls(h2oServer)[, 1])   

#Shutdown h20 instance
h2o.shutdown(h2oServer, prompt = FALSE)

#Make a submission file
write.csv(predictionNNFeatValidation, file = "NNBestFeatValidation.csv", row.names = FALSE)

#Model Ensemble-------------------
#Load Validation Predictions
#Start h2o from command line
system(paste0("java -Xmx5G -jar ", h2o.jarLoc, " -port 54333 -name Otto &"))
#Small pause
Sys.sleep(3)
#Connect R to h2o
h2oServer <- h2o.init(ip = "localhost", port = 54333, nthreads = -1)

#R data.table to h2o.ai
h2oOttoValidRF <- fread(file.path(workingDirectory, "RFValidation.csv"))
h2oOttoValidGBM <- fread(file.path(workingDirectory, "GBMValidation.csv"))
h2oOttoValidNN <- fread(file.path(workingDirectory, " NNValidation.csv"))
h2oOttoValidNNBestFeat <- fread(file.path(workingDirectory, "NNBestFeatValidation.csv"))
#True targets
h2oOttoTrain <- fread(file.path(dataDirectory, "train.csv"))

h2oOttoTrainValid <- as.h2o(cbind(h2oOttoValidRF, h2oOttoValidGBM, h2oOttoValidNN, h2oOttoValidNNBestFeat,
                                  h2oOttoTrain[c(dataSplits[[3]], dataSplits[[4]]), ncol(h2oOttoTrain)]))

#Print the approximate Logaritmic Loss of each model
#meanLogLoss 

#Make a gbm classifier
#h2o.ai Cross Validation
ensembleGBMCV <- h2o.gbm(x = seq(1, ncol(h2oOttoTrainValid)), y = ncol(h2oOttoTrainValid),
                         data = h2oOttoTrainValid[dataSplits[[3]], ],
                         nfolds = 5,
                         distribution = "multinomial",
                         interaction.depth = c(3, 5, 7),
                         shrinkage = c(0.001, 0.003), 
                         n.trees = 500,                           
                         grid.parallelism = 4)

#Log Info
mseGBMEnsemble <- mean(ensembleGBMCV@model[[1]]@model$err)
interaction.depthGBMEnsemble <- ensembleGBMCV@model[[1]]@model$params$interaction
shrinkageGBMEnsemble <- ensembleGBMCV@model[[1]]@model$params$shrinkage

#h2o.ai GBM Modelling    
ensembleGBM <- h2o.gbm(x = seq(1, ncol(h2oOttoTrainValid)), y = ncol(h2oOttoTrainValid),
                       data = h2oOttoTrainValid[c(dataSplits[[3]], dataSplits[[4]]), ],
                       distribution = "multinomial",
                       interaction.depth = interaction.depthGBMEnsemble,
                       shrinkage = shrinkageGBMEnsemble, 
                       n.trees = 8000)

print(paste0("Ensemble GBM Model procesed")) 

#Save model
h2o.saveModel(ensembleGBM, dir = dataDirectory, name = "ensembleGBM", save_cv = FALSE, force = FALSE)
h2o.rm(object = h2oServer, keys = h2o.ls(h2oServer)[, 1])   

#Shutdown h20 instance
h2o.shutdown(h2oServer, prompt = FALSE)

#Test data---------------------------------------
#Load Validation Predictions
#Start h2o from command line
system(paste0("java -Xmx5G -jar ", h2o.jarLoc, " -port 54333 -name Otto &"))
#Small pause
Sys.sleep(3)
#Connect R to h2o
h2oServer <- h2o.init(ip = "localhost", port = 54333, nthreads = -1)

#Read test data 
h2oOttoTest <- h2o.importFile(h2oServer, file.path(dataDirectory, "test.csv"))

##Load Models
h2o.loadModel(h2oServer, path = file.path(dataDirectory, "ModelRF"))
h2o.loadModel(h2oServer, path = file.path(dataDirectory, "ModelGBM"))
h2o.loadModel(h2oServer, path = file.path(dataDirectory, "ModelNN"))
h2o.loadModel(h2oServer, path = file.path(dataDirectory, "ModelNNBestFeat"))
h2o.loadModel(h2oServer, path = file.path(dataDirectory, "ensembleGBM"))

#Predictions calculations
testRF <- as.data.frame(h2o.predict(ottoRFModel, newdata = h2oOttoTest)[, seq(2, 10)])
print(paste0("RFs predictions ready")) 
testGBM <- as.data.frame(h2o.predict(ottoGBMModel, newdata = h2oOttoTest)[, seq(2, 10)])
print(paste0("GBM predictions ready")) 
testNN <- as.data.frame(h2o.predict(ottoDeepNetModel, newdata = h2oOttoTest)[, seq(2, 10)])
print(paste0("NNs predictions ready")) 
testNNBestFeat <- as.data.frame(h2o.predict(ottoDeepNetModelBestFeat, newdata = h2oOttoTest)[, seq(2, 10)])
print(paste0("NNBestFeats predictions ready")) 

#Ensemble calculation
h2oPredictions <- as.h2o(cbind(testRF, testGBM, testNN, testNNBestFeat))
ensemblePredictions <- as.data.frame(h2o.predict(ensembleGBM, newdata = h2oPredictions)[, seq(2, 10)])
print(paste0("NNBestFeats predictions ready"))

#Remove Data from server
h2o.rm(object = h2oServer, keys = h2o.ls(h2oServer)[, 1])
#Shutdown h20 instance
h2o.shutdown(h2oServer, prompt = FALSE)

#Make a submission file
sampleSubmissionFile <- fread(file.path(dataDirectory, "sampleSubmission.csv"), verbose = TRUE)
sampleSubmissionFile <- as.data.frame(sampleSubmissionFile)
sampleSubmissionFile[, seq(2, 10)] <- ensemblePredictions
write.csv(sampleSubmissionFile, file = "GBMEnsembleI.csv", row.names = FALSE)
system('zip GBMEnsembleI.zip GBMEnsembleI.csv')
