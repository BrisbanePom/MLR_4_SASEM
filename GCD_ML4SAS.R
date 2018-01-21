#Install the following packages and then load them
library(dplyr)
library(caret)
library(pacman)

#Read in functions scripts
source('GCD_ModelPlots.R')



#Utilise German Credit Data file from caret with pre-cleansed data
#Using Class as Binary (2 Level Factor) Target
data(GermanCredit)
df_GCD <- tbl_df(GermanCredit) %>%
    mutate(ID = row_number(),
           Class = factor(as.character(Class), levels=c("Good", "Bad"), ordered=TRUE)) %>%
    select(ID,Class,everything())

# Create a partition of X% of the rows for training, Y% for validation
# Note there is no validation in the traditional SAS EM manner;
#   instead CV training is used to optimise the model.
set.seed(42)
Partition_Index <- createDataPartition(df_GCD$Class,
                                       p=0.70, list=FALSE)
df_Input_Train <- df_GCD[Partition_Index,]
df_Input_HoldOut <- df_GCD[-Partition_Index,]

#Clean up interim objects
rm(GermanCredit, Partition_Index)

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
                  data=df_Input_Train,
                  method="glm",
                  family="binomial",
                  metric="ROC",
                  trControl=tc_GCD)
#warnings(GCD_Model) #Display detailed warnings


#Gather model predictions into evaluation data frame
dfEval_GLM <- ModelGather(model_GLM, 
                          df_Input_Train,
                          df_Input_HoldOut)

#Extract the evaluation results
plots_GLM <- ModelPlots(dfEval_GLM)
plots_GLM$ROC
plots_GLM$Lift
ModelVarImp(model_GLM)
summary(model_GLM)
print(model_GLM$finalModel)
#importance(model_GLM$finalModel)



#=== GLM Reduced variables =====================================================

#Train Logistic regression model (family = Binomial, method=glm) glmStepAIC
model_GLM2 <- train(Class ~ Duration+Amount+CheckingAccountStatus.lt.0+
                            Age+EmploymentDuration.4.to.7,
                   data=df_Input_Train,
                   method="glm",
                   family="binomial",
                   trControl=tc_GCD)
#warnings(GCD_Model) #Display detailed warnings


#Gather model predictions into evaluation data frame
dfEval_GLM2 <- ModelGather(model_GLM2, 
                          df_Input_Train,
                          df_Input_HoldOut)

#Extract the evaluation results
plots_GLM2 <- ModelPlots(dfEval_GLM2)
plots_GLM2$ROC
plots_GLM2$Lift
ModelVarImp(model_GLM2)



#=== GLMNET ===================================================================
#https://amunategui.github.io/binary-outcome-modeling/

model_GLMNet <- train(  Class ~ . -ID,
                        data = df_Input_Train,
                        method = "glmnet",
                        #tuneGrid = expand.grid(alpha = 0:1,
                        #                       lambda = seq(0.0001, 1, length = 20)),
                        family = "binomial",
                        trControl = tc_GCD)


#Create model evaluation data frame
dfEval_GLMNet <- ModelGather(model_GLMNet, 
                             df_Input_Train,
                             df_Input_HoldOut)


#Extract the evaluation results
plots_GLMNet <- ModelPlots(dfEval_GLMNet)
plots_GLMNet$ROC
plots_GLMNet$Lift
#ModelVarImp(model_GLMNet) - Variable importance not available for GLMNet
summary(model_GLMNet)
#importance(model_GLMNet$finalModel)


#=== Decision Tree ===================================================================
model_Tree <- train(Class ~ . -ID,
                  data = df_Input_Train,
                  method="rpart",
                  trControl= tc_GCD,
                  metric = "ROC")

#Create model evaluation data frame
dfEval_Tree <- ModelGather(model_Tree, 
                         df_Input_Train,
                         df_Input_HoldOut)

#Extract the evaluation results
plots_Tree <- ModelPlots(dfEval_Tree)
plots_Tree$ROC
plots_Tree$Lift
#ModelVarImp(model_GLMNet) - Variable importance not available for GLMNet
summary(model_Tree)
plot(model_Tree)



#=== Random Forest ===================================================================
model_RF <- train(Class ~ . -ID,
                  data = df_Input_Train,
                  method="rf",
                  ntree=100,
                  importance=TRUE,
                  na.action=na.omit,
                  #tuneGrid = rf.Grid,
                  trControl= tc_GCD,
                  metric = "ROC")

#Create model evaluation data frame
dfEval_RF <- ModelGather(model_RF, 
                        df_Input_Train,
                        df_Input_HoldOut)

#Extract the evaluation results
plots_RF <- ModelPlots(dfEval_RF)
plots_RF$ROC
plots_RF$Lift
#ModelVarImp(model_RF) #- Variable importance not available for GLMNet
summary(model_RF)
importance(model_RF$finalModel)


#====================================================================================
#Compare the results of the different models
#http://www.kimberlycoffey.com/blog/2016/7/16/compare-multiple-caret-run-machine-learning-models

results <- resamples(list(GLM=model_GLM, GLM2=model_GLM2,
                          GLMNet=model_GLMNet, RF=model_RF,
                          Tree=model_Tree))
# summarize the distributions
summary(results)
# boxplots of results
bwplot(results, metric ="Spec")
# dot plots of results
dotplot(results)
summary(model_GLMNet)
confusionMatrix(model_GLM2)






