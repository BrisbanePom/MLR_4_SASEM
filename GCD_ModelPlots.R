#Use ROCR to extract ROC and Lift plots and send to ggplot
library(ROCR)
library(ggplot2)


#Generate the Plot objects for the training partition
DF_Plot_Tr <- GCD_Evaluate %>%
    filter(Partition=="Training") %>% 
    mutate(bin = ntile(Pred_Prob, 20)) %>% 
    arrange(desc(bin)) %>% 
    select(Partition,ID,bin,everything())

PredObj_Train <- prediction(predictions = DF_Plot_Tr$bin,
                            labels      = DF_Plot_Tr$Observed)
ROC_Train <- performance(PredObj_Train, measure="tpr", x.measure="fpr")
Lift_Train <- performance(PredObj_Train, measure="lift", x.measure="rpp")
AUC_Train = round(100*performance(PredObj_Train ,"auc")@y.values[[1]],1)

#Generate the Plot objects for the validation partition
DF_Plot_Val <- GCD_Evaluate %>%
    filter(Partition=="Validation") %>% 
    mutate(bin = ntile(Pred_Prob, 20)) %>% 
    arrange(desc(bin)) %>% 
    select(Partition,ID,bin,everything())


PredObj_Valid <- prediction(predictions = DF_Plot_Val$bin,
                            labels      = DF_Plot_Val$Observed)
ROC_Valid <- performance(PredObj_Valid, measure="tpr", x.measure="fpr")
Lift_Valid <- performance(PredObj_Valid, measure="lift", x.measure="rpp")
AUC_Valid = round(100*performance(PredObj_Valid ,"auc")@y.values[[1]],1)


#Combine plot evaluation results into a single data frame
Evaluate_Results_DF <- bind_rows(
    data.frame(Partition="Training",
               ROC_X=unlist(ROC_Train@x.values),
               Lift_X=unlist(Lift_Train@x.values),
               ROC_Y=unlist(ROC_Train@y.values),
               Lift_Y=unlist(Lift_Train@y.values))
    ,
    data.frame(Partition="Validation",
               ROC_X=unlist(ROC_Valid@x.values),
               Lift_X=unlist(Lift_Valid@x.values),
               ROC_Y=unlist(ROC_Valid@y.values),
               Lift_Y=unlist(Lift_Valid@y.values)) 
)




#Plot ROC to GGPlot
ROC_Plot <- ggplot(data=Evaluate_Results_DF,aes(x=ROC_X,y=ROC_Y,col=Partition)) +
    geom_line(size=1.1) +
    scale_color_manual(values=c("#0000FF", "#FF0000")) +
    geom_abline(intercept = 0, slope = 1, color='grey') +
    annotate("text", x = .742, y = .25,
             label = paste0('AUC (Training): ', AUC_Train, '%'), size = 4) +
    annotate("text", x = .75, y = .18,
             label = paste0('AUC (Validation): ', AUC_Valid, '%'), size = 4) +
    labs(title = 'ROC Chart', y = 'Sensitivity (%)', x = 'Specificity (%)') +
    theme_minimal()
ROC_Plot


#Plot Lift to GGPlot
Lift_Plot <- ggplot(data=Evaluate_Results_DF,aes(x=Lift_X,y=Lift_Y,col=Partition)) +
    geom_line(size=1.1) +
    scale_color_manual(values=c("#0000FF", "#FF0000")) +
    geom_abline(intercept = 1, slope = 0, color='grey') +
    labs(title = 'Lift Chart', y = 'Percentile', x = 'Lift') +
    theme_minimal()
Lift_Plot

#Confusion Matrix on Validation data
CM <- confusionMatrix(data = DF_Plot_Val$Predicted,
                      reference = DF_Plot_Val$Observed)
CM


#Remove interim objects
rm(AUC_Train, AUC_Valid, Lift_Train, Lift_Valid, ROC_Train, ROC_Valid, PredObj_Train, PredObj_Valid)


