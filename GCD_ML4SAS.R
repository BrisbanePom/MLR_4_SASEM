#Install packages
library(dplyr)
library(caret)
library(pacman)


#Utilise German Credit Data file from caret with pre-cleansed data
#Using Class as Binary (2 Level Factor) Target
data(GermanCredit)
GCD <- tbl_df(GermanCredit) %>%
    mutate(ID = row_number(),
           Class = factor(as.character(Class), levels=c("Good", "Bad"), ordered=TRUE)) %>%
    select(ID,Class,everything())

# Create a partition of X% of the rows for training, Y% for validation
# Note there is no validation in the traditional SAS EM manner;
#   instead CV training is used to optimise the model.
set.seed(42)
Partition_Index <- createDataPartition(GCD$Class,
                                       p=0.70, list=FALSE)
GCD_Training <- GCD[Partition_Index,]
GCD_Validation <- GCD[-Partition_Index,]


#Build a train control object
tc_GCD <- trainControl (   method="cv", number = 10,
                           summaryFunction = twoClassSummary,
                           classProbs = TRUE,
                           verboseIter = TRUE,
                           savePredictions = TRUE)
#Add in pre-processing options: imputation and normalisation



#=== Basic GLM =================================================================

#Train Logistic regression model (family = Binomial, method=glm) glmStepAIC
model_GLM <- train(Class ~ . -ID,
                  data=GCD_Training,
                  method="glm",
                  #method="glmStepAIC", direction = "forward", steps=10,
                  family="binomial",
                  trControl=tc_GCD)
#warnings(GCD_Model) #Display detailed warnings


#Apply model to validation data set - Logistic Regression
GCD_Predict <-predict(model_GLM, GCD_Validation, type="prob")


#Create model evaluation data frame
dfEval_GLM <- ModelGather(model_GLM, GCD_Training,
                          GCD_Validation, GCD_Predict)


#Clean up interim objects
rm(GermanCredit, Partition_Index, GCD_Predict)


#Extract the evaluation plots
plots_GLM <- ModelPlots(dfEval_GLM)
plots_GLM$ROC
plots_GLM$Lift
ModelVarImp(model_GLM)
summary(model_GLM)


#=== GLMNET =================================================================
#https://amunategui.github.io/binary-outcome-modeling/

#Train Logistic regression model (family = Binomial, method=glm) glmStepAIC
model_GLMNet <- train(  Class ~ . -ID,
                        data=GCD_Training,
                        method="glmnet",
                        #tuneGrid = expand.grid(alpha = 0:1,
                        #                       lambda = seq(0.0001, 1, length = 20)),
                        family="binomial",
                        trControl=tc_GCD)


#Apply model to validation data set - Logistic Regression
GCD_Predict <-predict(model_GLMNet, GCD_Validation, type="prob")


#Create model evaluation data frame
dfEval_GLMNet <- ModelGather(model_GLMNet, GCD_Training,
                          GCD_Validation, GCD_Predict)

#Issue of Multiple iterations in data and having to filter by mtry


ModelVarImp(model_GLMNet)

results <- resamples(list(GLM=model_GLM, GLMNet=model_GLMNet))
# summarize the distributions
summary(results)
# boxplots of results
bwplot(results)
# dot plots of results
dotplot(results)
summary(model_GLMNet)






