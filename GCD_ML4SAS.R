#Install packages
library(dplyr)
library(caret)
#library(e1071)
library(pROC)
library(caTools)


#Utilise German Credit Data file from caret with pre-cleansed data 
#Set TARGET - BAD to 1 - i.e. we are going to predict defaults as the outcome
data(GermanCredit)
GCD <- tbl_df(GermanCredit) %>%
  mutate(ID = row_number(),
         Target = ifelse(Class == "Bad",1,0)) %>%
  select(ID,Class,Target,everything())

# Create a partition of 80% of the rows for training, 20% for validation
# Note there is no validation in the traditional SAS EM manner; instead CV training is used to optimise model
set.seed(42)
Partition_Index <- createDataPartition(GCD$Target,
                                       p=0.80, list=FALSE)
GCD_Training <- GCD[Partition_Index,]
GCD_Validation <- GCD[-Partition_Index,]


#(Check relative proportions of the target variables in the split data)
prop.table(table(GCD_Training$Target))
prop.table(table(GCD_Validation$Target))


#Build a train control object
GCD_TrCtrl <- trainControl(method="cv", 
                           summaryFunction=defaultSummary, 
                           classProbs=T,
                           savePredictions = T)
#Add in pre-processing: imputation and normalisation

#Train Logistic regression model (family = Binomial, method=glm) glmStepAIC
GCD_Model_LR <- train(Target ~
                     . -Class -ID,
                   data=GCD_Training,
                   method="glm",
                   family="binomial",
                   trControl=GCD_TrCtrl)
warnings(GCD_Model_LR)

# # #Train decision tree model (family = Binomial, method=glm) glmStepAIC
# GCD_Model_DT <- train(Target ~
#                         . -Class -ID,
#                       data=GCD_Training,
#                       method="rpart2",
#                       family="binomial",
#                       metric='RMSE',
#                       trControl=GCD_TrCtrl)
# warnings(GCD_Model_DT)


#Show model variable importance
predictors(GCD_Model_LR)
varImp(GCD_Model_LR)
plot(varImp(GCD_Model_LR))



#Aply model to validation data set - Logistic Regression
GCD_Pred_LR.obj <-predict(GCD_Model_LR, GCD_Validation)
GCD_Predict_LR <- GCD_Validation %>%
  select(ID,Observed=Target) %>%
  mutate(Predicted = ifelse(GCD_Pred_LR.obj > 0.5, 1,0),
         Pred_Prob = GCD_Pred_LR.obj) %>%
  arrange(desc(Pred_Prob))

#ConfusionMatrix on Validation data
confusionMatrix(data = GCD_Predict_LR$Predicted,
                reference = GCD_Predict_LR$Observed)

# #Aply model to validation data set - Decision Tree
# GCD_Pred_DT.obj <-predict(GCD_Model_DT, GCD_Validation)
# GCD_Predict_DT <- GCD_Validation %>%
#   select(ID,Observed=Target) %>%
#   mutate(Predicted = ifelse(GCD_Pred_DT.obj > 0.5, 1,0),
#          Pred_Prob = GCD_Pred_DT.obj) %>%
#   arrange(desc(Pred_Prob))
# 
# #ConfusionMatrix on Validation data
# confusionMatrix(data = GCD_Predict_LR$Predicted,
#                 reference = GCD_Predict_LR$Observed)



#Output AUC values
ROC.Training_LR <- roc(GCD_Model_LR$pred$obs, GCD_Model_LR$pred$pred)
ROC.Validation_LR <- roc(GCD_Predict_LR$Observed, GCD_Predict_LR$Pred_Prob)
print("TRAINING"); auc(ROC.Training_LR)
print("VALIDATION"); auc(ROC.Validation_LR)

### Plotting ROC curves

# a) using base graphics system

#Plot ROC curves
plot(ROC.Training_LR, col = "blue",
     print.auc = TRUE, print.auc.y = .4)
plot(ROC.Validation_LR, col = "magenta",
     print.auc = TRUE, print.auc.y = .2,
     add = TRUE)

# b) using ggplot2 graphics system

library(ggplot2)

# can be extended to accommodate comparison of ROC curves, and 
# certainly could stand some aesthetic tweaks... but a gr8 start!
ggplotROC = function(roc, colour="black", title = NULL){
    # for ggplot solution: h/t @ https://stackoverflow.com/questions/37438461/
    auc_percent = round(roc$auc * 100, 2)
    df = data.frame(Specificity=roc$specificities, Sensitivity=roc$sensitivities)
    plot = ggplot(data = df, aes(x = Specificity, y = Sensitivity))+
        geom_path(colour = colour, size = 1.5)+
        scale_x_reverse() +
        geom_abline(intercept = 1, slope = 1, color='grey')+
        annotate("text", x = .85, y = .95, 
            label = paste0('AUC: ', auc_percent, '%'), size = 5) +
        labs(title = title, y = 'Sensitivity (%)', x = 'Specificity (%)')
    # adding a clean and simple theme
    plot = plot + theme_minimal()
    return(plot)
}

# plots created from 
trainRocPlot = ggplotROC(ROC.Training_LR, colour = "red",
    title = "Training Set Performance")

testRocPlot = ggplotROC(ROC.Validation_LR, colour = "blue",
    "Validation Set Performance")


