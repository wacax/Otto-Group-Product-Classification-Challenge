#AXA Driver Telematics Analysis
#Ver 0.1 #Otto Group Product Classification Challenge RF first try

#Init-----------------------------------------------
rm(list=ls(all=TRUE))

#Libraries, directories, options and extra functions----------------------
require("data.table")
require("h2o")

#Set Working Directory
workingDirectory <- "/home/wacax/Wacax/Kaggle/Otto/"
setwd(workingDirectory)
dataDirectory <- "/home/wacax/Wacax/Kaggle/Otto/Data/"
#h2o location
h2o.jarLoc <- "/home/wacax/R/x86_64-pc-linux-gnu-library/3.1/h2o/java/h2o.jar"

#Load data
train <- fread(file.path(dataDirectory, "train.csv"), verbose = TRUE)
test <- fread(file.path(dataDirectory, "test.csv"), verbose = TRUE)

#EDA----------------------------------


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
                                  depth = c(20, 50, 75), 
                                  verbose = FALSE)

#Log Info
aucRF <- ottoRFModelCV@model[[1]]@model$auc
ntreeRF <- ottoRFModelCV@model[[1]]@model$params$ntree
depthRF <- ottoRFModelCV@model[[1]]@model$params$depth

#h2o.ai RF Modelling    
ottoRFModel <- h2o.randomForest(x = seq(2, ncol(h2oOttoTrain)), y = ncol(h2oOttoTrain),
                                data = h2oOttoTrain[randIdxs, ],
                                nfolds = 5,
                                classification = TRUE,
                                ntree = ntreeRF,
                                depth = depthRF, 
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

write.csv(sampleSubmissionFile, file = "RFPredictionI.csv", row.names = FALSE)
system('zip RFPredictionI.zip RFPredictionI.csv')

