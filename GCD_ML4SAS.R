#Install packages
library(dplyr)
library(caret)
library(e1071)

#Utilise German Credit Data file from caret with pre-cleansed data 
#Set TARGET - BAD to 1 - i.e. we are going to predict defaults as the outcome
data(GermanCredit)
GCD <- tbl_df(GermanCredit) %>%
  mutate(ID = row_number(),
         Target = ifelse(Class == "Bad",1,0)) %>%
  select(ID,Class,Target,everything())

# Create a partition of 60% of the rows for training, 40% for validation
set.seed(42)
Partition_Index <- createDataPartition(GCD$Target,
                                       p=0.60, list=FALSE)
GCD_Training <- GCD[Partition_Index,]
GCD_Validation <- GCD[-Partition_Index,]

#Check relative proportions of the target variables
prop.table(table(GCD_Training$Target))
prop.table(table(GCD_Validation$Target))


#Train Logistic regression model (family = Binomial, method=glm)
GCD_Model <- train(Target ~
                     . -Class,
                   data=GCD_Training,
                   method="glm",
                   family="binomial")

#Show model variable importance
plot(varImp(GCD_Model))


#Aply model to validation data
GCD_Predict <- GCD_Validation %>%
  select(ID,Class,Target) %>%
  mutate(Predicted = predict(GCD_Model, GCD_Validation),
         Conf_Pred = ifelse(Predicted > 0.5, 1,0)) %>%
  arrange(desc(Predicted))

#ConfusionMatrix on Validation data
confusionMatrix(data = GCD_Predict$Conf_Pred,
                reference = GCD_Predict$Target)
