#Install packages
library(dplyr)
library(caret)


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


#(Check relative proportions of the target variables in the split data)
prop.table(table(GCD_Training$Class))
prop.table(table(GCD_Validation$Class))


#Build a train control object
GCD_TrCtrl <- trainControl(method="cv",
                           summaryFunction=defaultSummary,
                           classProbs=T,
                           savePredictions = T)
#Add in pre-processing: imputation and normalisation

#Train Logistic regression model (family = Binomial, method=glm) glmStepAIC
GCD_Model_LR <- train(Class ~ . -ID,
                      data=GCD_Training,
                      method="glm",
                      family="binomial",
                      trControl=GCD_TrCtrl)
warnings(GCD_Model_LR) #Display detailed warnings


#Apply model to validation data set - Logistic Regression
GCD_Pred_LR <-predict(GCD_Model_LR,
                      GCD_Validation,
                      type="prob")


#Compile training and validation data into a single data frame for evaluation
#detach(package:MASS) #(Detach MASS package if called by caret to avoid conflict with dplyr)
GCD_Evaluate_Tr <- GCD_Model_LR$pred %>%
    select(ID=rowIndex, Predicted=pred, Observed=obs, Pred_Prob=Bad) %>%
    mutate(Partition="Training") %>%
    select(Partition,ID,Pred_Prob,Predicted,Observed) %>%
    arrange(desc(Pred_Prob))

GCD_Evaluate_Val <- GCD_Validation %>%
    select(ID, Observed=Class) %>%
    mutate(Partition="Validation",
           Pred_Prob = GCD_Pred_LR$Bad,
           Predicted = factor(as.character(ifelse(GCD_Pred_LR$Bad > 0.5,"Bad","Good")),
                              levels=c("Good", "Bad"), ordered=TRUE)) %>% 
    select(Partition,ID,Pred_Prob,Predicted,Observed) %>%
    arrange(desc(Pred_Prob))


GCD_Evaluate <- bind_rows(GCD_Evaluate_Tr,GCD_Evaluate_Val)  



#Clean up interim objects
rm(GermanCredit,Partition_Index,GCD_Pred_LR, GCD_Evaluate_Tr, GCD_Evaluate_Val)




#WIP - to develop / migrate into evaluation function
#===================================================


#Show model variable importance - move into evaluation / plots code
predictors(GCD_Model_LR)
VI1 <- varImp(GCD_Model_LR)
plot(VI1)






