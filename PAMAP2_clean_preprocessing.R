setwd("C:/Users/alexh/Desktop/PAMAP2_Dataset/Protocol")
setwd("D:/ADS/PAMAP2_Dataset/Protocol")

formatDataset <- function(datasetPath, subjectInfo = NULL) {
        tempData <- read.table(datasetPath, header = F, sep = " ")
        
        names(tempData) <- c("timeStamp", "activity", "HR", 
                             paste0("hand_", c("temp", "acce_x", "acce_y", "acce_z", 
                                               "acce_x_2", "acce_y_2", "acce_z_2", 
                                               "gyro_x", "gyro_y", "gyro_z",
                                               "magn_x", "magn_y", "magn_z",
                                               "orie_1", "orie_2", "orie_3", "orie_4")),
                             paste0("chest_", c("temp", "acce_x", "acce_y", "acce_z", 
                                                "acce_x_2", "acce_y_2", "acce_z_2", 
                                                "gyro_x", "gyro_y", "gyro_z",
                                                "magn_x", "magn_y", "magn_z",
                                                "orie_1", "orie_2", "orie_3", "orie_4")),
                             paste0("ankle_", c("temp", "acce_x", "acce_y", "acce_z", 
                                                "acce_x_2", "acce_y_2", "acce_z_2", 
                                                "gyro_x", "gyro_y", "gyro_z",
                                                "magn_x", "magn_y", "magn_z",
                                                "orie_1", "orie_2", "orie_3", "orie_4")))
        # tempData <- tempData[tempData$activity != 0,]
        tempData <- tempData[,c(1:3, 4:7, 11:16, 21:24, 28:33, 38:41, 45:50)]
        
        for (idx in 1:ncol(tempData)) {
                tempData[[idx]][is.nan(tempData[[idx]])] <- NA
        }
        
        if (!is.null(subjectInfo)) {
                tempData <- cbind(t(data.frame(subjectInfo)), tempData)
        }
        
        tempData
}

dataset1 <- formatDataset("subject101.dat", 
                          subjectInfo = c(num = 1, sex = 1, age = 27, height = 182, weight = 83, 
                                          restingHR = 75, maxHR = 193, hand = 1))
dataset2 <- formatDataset("subject102.dat", 
                          subjectInfo = c(num = 2, sex = 0, age = 25, height = 169, weight = 78, 
                                          restingHR = 74, maxHR = 195, hand = 1))
dataset3 <- formatDataset("subject103.dat", 
                          subjectInfo = c(num = 3, sex = 1, age = 31, height = 187, weight = 92, 
                                          restingHR = 68, maxHR = 189, hand = 1))
dataset4 <- formatDataset("subject104.dat", 
                          subjectInfo = c(num = 4, sex = 1, age = 24, height = 194, weight = 95, 
                                          restingHR = 58, maxHR = 196, hand = 1))
dataset5 <- formatDataset("subject105.dat", 
                          subjectInfo = c(num = 5, sex = 1, age = 26, height = 180, weight = 73, 
                                          restingHR = 70, maxHR = 194, hand = 1))
dataset6 <- formatDataset("subject106.dat", 
                          subjectInfo = c(num = 6, sex = 1, age = 26, height = 183, weight = 69, 
                                          restingHR = 60, maxHR = 194, hand = 1))
dataset7 <- formatDataset("subject107.dat", 
                          subjectInfo = c(num = 7, sex = 1, age = 23, height = 173, weight = 86, 
                                          restingHR = 60, maxHR = 197, hand = 1))
dataset8 <- formatDataset("subject108.dat", 
                          subjectInfo = c(num = 8, sex = 1, age = 32, height = 179, weight = 87, 
                                          restingHR = 66, maxHR = 188, hand = 0))
dataset9 <- formatDataset("subject109.dat", 
                          subjectInfo = c(num = 9, sex = 1, age = 31, height = 168, weight = 65, 
                                          restingHR = 54, maxHR = 189, hand = 1))

randomForest_CV <- function(datasets = list(), label.col = 1,
                            positive.class = NULL, alpha.range = seq(0, 1, 0.2),
                            folds.num = 10, seed = 1, parallel.cores = 10) {
        message("+ Initializing...  ", Sys.time())
        
        all_folds <- lapply(datasets, function(x) {
                set.seed(seed)
                folds <- caret::createFolds(x$num, k = folds.num, returnTrain = TRUE)
        })
        
        parallel.cores <- ifelse(parallel.cores == -1, parallel::detectCores(), parallel.cores)
        parallel.cores <- ifelse(parallel.cores > folds.num, folds.num, parallel.cores)
        
        if(parallel.cores == 2 & folds.num > parallel.cores) {
                message("Users can try to set parallel.cores = -1 to use all cores!", "\n")
        }
        
        cl <- parallel::makeCluster(parallel.cores)
        
        if(is.null(positive.class)) {
                positive.class <- as.character(datasets[[1]]$label[[1]])
        }
        parallel::clusterExport(cl, varlist = c("datasets", "all_folds", "positive.class", "ntree"), envir = environment())
        
        message("\n+ ", folds.num, "-fold CV Processing...\n")
        perf.res <- parallel::parSapply(cl, 1:folds.num, function(x) {
                trainSet <- c()
                testSet <- c()
                for(i in 1:length(datasets)) {
                        dataset_tmp <- datasets[[i]]
                        fold_tmp <- all_folds[[i]]
                        train_tmp <- dataset_tmp[fold_tmp[[x]],]
                        test_tmp <- dataset_tmp[-fold_tmp[[x]],]
                        
                        trainSet <- rbind(trainSet, train_tmp)
                        testSet <- rbind(testSet, test_tmp)
                }
                
                LR_mod <- randomForest::randomForest(label ~ ., data = trainSet,
                                                     ntree = ntree, importance = TRUE)
                res <- predict(RF_mod, testSet, type = "response")
                
                confusion.res <- caret::confusionMatrix(data.frame(res)[,1], testSet$label,
                                                        positive = positive.class,
                                                        mode = "everything")
                
                performance.res <- data.frame(Sensitivity = confusion.res$byClass[1],
                                              Specificity = confusion.res$byClass[2],
                                              Accuracy    = confusion.res$overall[1],
                                              F.Measure   = confusion.res$byClass[7],
                                              Kappa       = confusion.res$overall[2])
                
        })
        
        parallel::stopCluster(cl)
        Ave.res <- apply(perf.res, 1, as.numeric)
        Ave.res <- as.data.frame(t(Ave.res))
        Ave.res$Ave.Res <- rowMeans(Ave.res)
        message("- Performance:")
        print(round(t(Ave.res[ncol(Ave.res)]), digits = 4))
        message("\n+ Completed.   ", Sys.time())
        Ave.res
}

library(ggplot2)

ggplot(data = data_test, aes(x = 1:length(deltaData), y = deltaData)) +
        geom_line(size = 0.2, alpha = I(0.5)) + geom_point(size = 1.6, alpha = I(0.8)) +
        scale_x_reverse(breaks = seq(5, 29, 2)) +
        scale_y_continuous(breaks = seq(0.8, 0.9, 0.02)) +
        coord_cartesian(xlim = c(6.8, 28.2)) +
        guides(fill = guide_legend(nrow = 1, byrow = TRUE))


# newTemp <- tempData[tempData$activity != 0,]
# 
# allDataset <- allDataset[allDataset$num != 9,]
# allDatasetCleaned <- allDataset[allDataset$activity != 0,]
# 
# allDatasetCleaned$activity[allDatasetCleaned$activity == 1] <- "lying"
# allDatasetCleaned$activity[allDatasetCleaned$activity == 2] <- "sitting"
# allDatasetCleaned$activity[allDatasetCleaned$activity == 3] <- "standing"
# allDatasetCleaned$activity[allDatasetCleaned$activity == 4] <- "walking"
# allDatasetCleaned$activity[allDatasetCleaned$activity == 5] <- "running"
# allDatasetCleaned$activity[allDatasetCleaned$activity == 6] <- "cycling"
# allDatasetCleaned$activity[allDatasetCleaned$activity == 7] <- "NordicWalking"
# allDatasetCleaned$activity[allDatasetCleaned$activity == 12] <- "ascendingStairs"
# allDatasetCleaned$activity[allDatasetCleaned$activity == 13] <- "descendingStairs"
# allDatasetCleaned$activity[allDatasetCleaned$activity == 16] <- "vacuumCleaning"
# allDatasetCleaned$activity[allDatasetCleaned$activity == 17] <- "ironing"
# allDatasetCleaned$activity[allDatasetCleaned$activity == 24] <- "ropeJumping"

# table(allDatasetCleaned$activity)
# 
# save(allDatasetCleaned, file = "allDatasetCleaned.RData")
# 
inputDataset <- allDatasetCleaned
function(inputDataset) {
        inputLabel <- inputDataset$activity
        label1 <- inputLabel[-1]
        label2 <- inputLabel[-length(inputLabel)]
        labelPasted <- paste(label2, label1, sep = ".")
        labelCount <- table(labelPasted)
        datasetTemp <- data.frame(labelCount, stringsAsFactors = F)
        datasetTemp$labelPasted <- as.character(datasetTemp$labelPasted)
        outputVal <- sapply(datasetTemp$labelPasted, function(x) {
                val <- strsplit(x, ".", fixed = T)[[1]]
                val
        })
        outputVal <- t(outputVal)
        outputVal.df <- data.frame(outputVal, datasetTemp$Freq, row.names = NULL)
        names(outputVal.df) <- c("current", "next", "freq")
        outputVal.df
}

write.table(outputVal.df, file = "activityChanges.csv", sep = ",", col.names = F)

function1 <- function(inputDataset) {
        propNAs <- sapply(1:ncol(inputDataset), function(x) {
                selectedData <- inputDataset[[x]]
                propNA <- which(is.na(selectedData))
                if(length(propNA) == 0) propNA <- 0
                propNA
        })
        res <- table(rowMeans(propNAs) == propNAs[,1])
        res
}

inputDataset <- dataset3
function2 <- function(inputDataset) {
        # message("hand:")
        rowNum <- nrow(inputDataset)
        numTmp <- function1(inputDataset[12:21])
        cat(numTmp, round(numTmp / rowNum * 100, 2), sep = " & ")
        # message("chest:")
        cat(" & ")
        numTmp <- function1(inputDataset[22:31])
        cat(numTmp, round(numTmp / rowNum * 100, 2), sep = " & ")
        cat(" & ")
        # message("ankle:")
        numTmp <- function1(inputDataset[32:41])
        cat(numTmp, round(numTmp / rowNum * 100, 2), sep = " & ")
        cat("\n")
}

for(i in ls(pattern = "dataset")) {
        inputDataset <- get(i)
        function2(inputDataset)
}

checkNaProp <- function(inputDataset) {
        propNAs <- sapply(1:ncol(inputDataset), function(x) {
                selectedData <- inputDataset[[x]]
                propNA <- which(is.na(selectedData)) / nrow(inputDataset)
                if(length(propNA) == 0) propNA <- 0
                propNA
        })
        currVal <- data.frame(Freq = propNAs[[1]], IMU = strsplit(names(inputDataset)[[1]], "_")[[1]][1])
        if(length(propNAs) > 1) {
                for(i in 2:length(propNAs)) {
                        currValTmp <- data.frame(Freq = propNAs[[i]], IMU = strsplit(names(inputDataset)[[i]], "_")[[1]][1])
                        currVal <- rbind(currVal, currValTmp)
                }
        }
        currVal
}

makeNaProp <- function(datasetList = ls(pattern = "dataset")) {
        outputNaProp <- data.frame()
        for(i in datasetList) {
                inputDataset <- get(i)
                inputDatasetNAProp <- checkNaProp(inputDataset[,c(15, 22, 36)])
                inputDatasetNAProp <- cbind(inputDatasetNAProp, Group = paste0("subject", seqinr::s2c(i)[8]))
                outputNaProp <- rbind(outputNaProp, inputDatasetNAProp)
        }
        outputNaProp
}

NaProp <- makeNaProp(datasetList = ls(pattern = "dataset"))

NaProp$label <- as.factor(paste(NaProp$Group, NaProp$IMU, sep = "_"))
class(NaProp$label)
save(NaProp, file = "NA_posDist.RData")

library(ggplot2)

ggplot2::ggplot(data = NaProp, aes(x = label, y = Freq)) +
        geom_jitter(alpha = I(0.1), aes(colour = Group), height = 0) +
        geom_boxplot(alpha = I(0.6), aes(fill = Group)) +
        scale_y_continuous(breaks = c(seq(0, 1, 0.1))) +
        scale_x_discrete(label = rep(c("Hand", "Chest", "Ankle"), 8)) +
        theme(axis.title.x = element_text(size = 12),
              axis.text.x  = element_text(size = 12, angle = -45),
              axis.title.y = element_text(size = 12),
              axis.text.y  = element_text(size = 12),
              legend.title = element_text(size = 12),
              legend.text  = element_text(size = 12),
              legend.position = "bottom") +
        guides(colour = guide_legend(nrow = 1)) # 12 * 7 in



getSubset <- function(inputDataset, parallel.cores = 20) {
        cl <- parallel::makeCluster(parallel.cores)
        parallel::clusterExport(cl, varlist = "inputDataset", envir = environment())
        datasetIdx <- parallel::parSapply(cl, 1:nrow(inputDataset), function(x) {
                flagNA <- !anyNA(inputDataset[x,])
                flagNA
        })
        parallel::stopCluster(cl)
        outputDataset <- inputDataset[datasetIdx,]
        outputDataset
}

for (i in ls(pattern = "dataset")) {
        message("Processing Dataset: ", i, "   ", Sys.time())
        inputDataset <- get(i)
        tmpDataset <- getSubset(inputDataset, 30)
        assign(paste0("subset_", i), tmpDataset)
}

subset <- getSubset(inputDataset, 8)

createNAs <- function(inputDataset, prop = 0.03, processIdx = 11:ncol(inputDataset)) {
        for (i in processIdx) {
                set.seed(i)
                idxNAs <- sample(nrow(inputDataset), floor(prop * nrow(inputDataset)))
                inputDataset[idxNAs,i] <- NA
        }
        inputDataset
}

imputeteDataset <- function(datasetName, method = "pmm") {
        datasetNow <- get(datasetName)
        miceMod <- mice::mice(datasetNow[,11:ncol(datasetNow)], method = method)
        datasetNew <- mice::complete(miceMod)
        datasetNew <- cbind(datasetNow[,1:10], datasetNew)
        message("Complete: ", datasetName)
        datasetNew
}

library(mice)

cl <- parallel::makeCluster(8, outfile = "")
parallel::clusterExport(cl, varlist = paste0("NA_subset2_dataset", 1:8))

imputateDatasets_pmm <- parallel::parLapply(cl, ls(pattern = "NA_subset2_dataset"), imputeteDataset, method = "pmm")
imputateDatasets_sample <- parallel::parLapply(cl, ls(pattern = "NA_subset2_dataset"), imputeteDataset, method = "sample")
imputateDatasets_mean <- parallel::parLapply(cl, ls(pattern = "NA_subset2_dataset"), imputeteDataset, method = "mean")

save(imputateDatasets_pmm, file = "imputedSubset_test_pmm.RData")
save(imputateDatasets_sample, file = "imputedSubset_test_sample.RData")
save(imputateDatasets_mean, file = "imputedSubset_test_mean.RData")


imputeLOCF <- function(datasetWithNA, datasetNoNA) {
        imputedDataset <- tidyr::fill(data = datasetWithNA, 1:ncol(datasetWithNA))
        imputedDataset <- tidyr::fill(data = imputedDataset, 1:ncol(datasetWithNA), .direction = "up")
        
        dataNA <- unlist(datasetWithNA)
        idxNAs <- which(is.na(dataNA))
        numNAs <- length(idxNAs)
        
        dataPred <- unlist(imputedDataset)[idxNAs]
        dataRef <- unlist(datasetNoNA)[idxNAs]
        
        dataTmp <- dataPred - dataRef
        
        RMSE <- sqrt(sum(dataTmp ^ 2)/numNAs)
        PBIAS <- abs(sum(dataTmp) / sum(dataRef)) * 100
        
        message("PBIAS: ", PBIAS, "; RMSE: ", RMSE)
        
        output <- list(c(PBIAS = PBIAS, RMSE = RMSE), imputedDataset = imputedDataset)
        output
}

for(i in 1:8) {
        datasetWithNA <- get(paste0("NA_subset2_dataset", i))
        datasetNoNA   <- get(paste0("subset2_dataset", i))
        
        tmpDataset <- imputeLOCF(datasetWithNA[,-11], datasetNoNA[,-11])
        
        assign(paste0("imputeDataset", i, "_LOCF"), tmpDataset)
}
rm(datasetWithNA, datasetNoNA, tmpDataset)
save(list = ls(pattern = "imputeData"), file = "imputedSubset_test_LOCF.RData")

evaluateImputation <- function(imputedDataset, datasetWithNA, datasetNoNA) {
        dataNA <- unlist(datasetWithNA)
        idxNAs <- which(is.na(dataNA))
        numNAs <- length(idxNAs)
        
        dataPred <- unlist(imputedDataset)
        dataRef <- unlist(datasetNoNA)
        
        dataTmp <- dataPred - dataRef
        
        RMSE <- sqrt(sum(dataTmp ^ 2)/numNAs)
        # PBIAS <- sum(abs(dataTmp / dataRef * 100))/numNAs
        RBIAS <- sum(abs(dataTmp))/numNAs
        
        message("RBIAS: ", round(RBIAS, 3), " & ", round(RMSE, 3), " :RMSE")
        
        output <- c(RBIAS = RBIAS, RMSE = RMSE)
        output
}

for(i in 1:8) {
        NA_subset2_dataset <- get(paste0("NA_subset2_dataset", i))
        subset2_dataset <- get(paste0("subset2_dataset", i))
        imputedDataset <- get(ls(pattern = "imputate"))
        res <- evaluateImputation(imputedDataset = imputedDataset[[i]][,12:41], 
                                  datasetWithNA = NA_subset2_dataset[,12:41], 
                                  datasetNoNA = subset2_dataset[,12:41])
}

interpolationNA <- function(NAdataset, Refdataset = NULL) {
        bakDataset <- NAdataset
        for(j in 1:length(NAdataset)){
                # cat(j, sep = " ")
                if(is.na(NAdataset[[j]][[1]])) NAdataset[[j]][[1]] <- NAdataset[[j]][which(!is.na(NAdataset[[j]]))[1]]
                if(is.na(tail(NAdataset[[j]], 1))) {
                        NAdataset[[j]][[tail(NAdataset[[j]], 1)]] <- NAdataset[[j]][rev(which(!is.na(NAdataset[[j]])))[1]]
                }
                NAdataset[[j]] <- zoo::na.approx(NAdataset[[j]])
        }
        
        if(is.null(Refdataset)) {
                return(NAdataset)
        } else {
                res <- evaluateImputation(NAdataset, bakDataset, Refdataset)
                return(res)
        }
}

csInterpolating <- function(inputDataset) {
        
        newDataset <- interpolationNA(inputDataset[,12:41])
        newDataset <- cbind(inputDataset[,1:11], newDataset)
        
        # subset <- getSubset(newDataset, 20)
        # NAdataset <- subset
        # NAdataset$HR[sample(nrow(NAdataset), floor(0.5 * nrow(NAdataset)))] <- NA
        # datasetWithNA <- NAdataset
        # 
        # if(is.na(NAdataset$HR[1])) NAdataset$HR[1] <- NAdataset$HR[which(!is.na(NAdataset$HR))[1]]
        # if(is.na(tail(NAdataset$HR, 1))) NAdataset$HR[length(NAdataset$HR)] <- NAdataset$HR[rev(which(!is.na(NAdataset$HR)))[1]]

        # NAdataset$HR[is.na(NAdataset$HR)] <- stats::spline(NAdataset$timeStamp, NAdataset$HR, n = length(NAdataset$HR), method = "natural")$y[is.na(NAdataset$HR)]
        
        # res <- evaluateImputation(imputedDataset = NAdataset$HR, datasetWithNA = datasetWithNA$HR, datasetNoNA = subset$HR)
        # print(res)
        # res
        idxNA <- is.na(newDataset$HR)
        newDataset$HR[idxNA] <- stats::spline(newDataset$timeStamp, 
                                              newDataset$HR, n = length(idxNA), 
                                              method = "natural")$y[idxNA]
        newDataset
}

for(i in 1:8){
        inputDataset <- get(paste0("dataset", i))
        imputedDataset <- csInterpolating(inputDataset)
        imputedDataset <- imputedDataset[imputedDataset$activity != 0,]
        write.csv(imputedDataset, file = paste0("imputedPAMAP2_", i, ".csv"))
        assign(paste0("imputedDataset_v3_", i), imputedDataset)
}

function(inputDataset) {
        inputDataset <- inputDataset[,-c(1:10)]
        inputDatasetRandom <- inputDataset[sample(nrow(inputDataset)),]
        idx <- round(seq(nrow(inputDatasetRandom), 1, length.out = 11))
        perf <- data.frame()
        for(i in 1:10) {
                testSetIDX <- idx[i]:idx[(i + 1)]
                testSet <- inputDatasetRandom[testSetIDX,]
                trainSet <- inputDatasetRandom[-testSetIDX,]
                
                model <- caret::train(
                        HR ~., data = trainSet, method = "glmnet",
                        trControl = caret::trainControl("cv", number = 10),
                        tuneLength = 10
                )
                
                Res <- predict(model, testSet)
                
                dataTmp <- Res - testSet$HR
                RMSE <- sqrt(sum(dataTmp ^ 2)/nrow(testSet))
                RBIAS <- sum(abs(dataTmp))/nrow(testSet)
                message("RBIAS: ", round(RBIAS, 3), " & ", round(RMSE, 3), " :RMSE")
                
                res <- data.frame(RMSE = RMSE, RBIAS = RBIAS)
                perf <- rbind(perf, res)
        }
        message("Final:")
        output.perf <- colMeans(perf)
        output.perf
}

for(num in 1:8) {
        inputDataset <- get(paste0("dataset_v2_", num))
        newDataset <- getSubset(inputDataset)
        assign(paste0("dataset_v2_NoNA_", num), newDataset)
        newDataset$HR[sample(nrow(newDataset), floor(0.5 * nrow(newDataset)))] <- NA
        assign(paste0("dataset_v2_WithNA_", num), newDataset)
}

runGLM <- function(inputDataset) {
        inputDataset <- inputDataset[,-c(1:10)]
        inputDatasetRandom <- inputDataset[sample(nrow(inputDataset)),]
        idx <- round(seq(nrow(inputDatasetRandom), 1, length.out = 11))
        perf <- data.frame()
        for(i in 1:10) {
                testSetIDX <- idx[i]:idx[(i + 1)]
                testSet <- inputDatasetRandom[testSetIDX,]
                trainSet <- inputDatasetRandom[-testSetIDX,]
                
                model <- caret::train(
                        HR ~., data = trainSet, method = "glmnet",
                        trControl = caret::trainControl("cv", number = 10),
                        tuneLength = 10
                )
                
                Res <- predict(model, testSet)
                
                dataTmp <- Res - testSet$HR
                RMSE <- sqrt(sum(dataTmp ^ 2)/nrow(testSet))
                RBIAS <- sum(abs(dataTmp))/nrow(testSet)
                message("RBIAS: ", round(RBIAS, 3), "RMSE: ", round(RMSE, 3))
                print(model$bestTune)
                
                res <- data.frame(RMSE = RMSE, RBIAS = RBIAS)
                perf <- rbind(perf, res)
        }
        message("Final:")
        output.perf <- colMeans(perf)
        print(output.perf)
        output.perf
}

for(i in 1:8) {
        message("Subj ", i)
        inputDataset <- get(paste0("dataset_v2_NoNA_", i))
        runGLM(inputDataset)
}
