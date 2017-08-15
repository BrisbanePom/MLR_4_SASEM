#Use ROCR to extract ROC and Lift plots and send to ggplot


#Build a prediction object on the Train data and generate ROC and Lift objects
PredObj_Train <- prediction(predictions = GCD_Evaluate_Tr$Pred_Prob,
                            labels      = GCD_Evaluate_Tr$Observed)
ROC_Train <- performance(PredObj_Train, measure="tpr", x.measure="fpr")
Lift_Train <- performance(PredObj_Train, measure="lift", x.measure="rpp")

#Build a prediction object on the Validation data and generate ROC and Lift objects
PredObj_Valid <- prediction(predictions = GCD_Evaluate_Val$Pred_Prob,
                            labels      = GCD_Evaluate_Val$Observed)
ROC_Valid <- performance(PredObj_Valid, measure="tpr", x.measure="fpr")
Lift_Valid <- performance(PredObj_Valid, measure="lift", x.measure="rpp")

#Combine evaluation results into a data frame
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


#Calculate the AUC values;
AUC_Train = round(100*performance(PredObj_Train ,"auc")@y.values[[1]],1)
AUC_Valid = round(100*performance(PredObj_Valid ,"auc")@y.values[[1]],1)



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





# #Default ROCR plots for comparison
# plot(ROC_Train,
#      main="Cross-Sell - ROC Curves",
#      xlab="1 – Specificity: False Positive Rate",
#      ylab="Sensitivity: True Positive Rate",
#      col="blue")
# abline(0,1,col="grey")
# plot(Lift_Train,
#      main="Cross-Sell - Lift Chart",
#      xlab="% Populations",
#      ylab="Lift",
#      col="blue")
# abline(1,0,col="grey")