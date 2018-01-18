#=========================================================================================|
#FUNCTION: Model Gather                                                                   |
#                                                                                         |
#Purpose:   Gather the validation and training data into the evaluation data frame        |
#Accepts:   Training data, validation data, model                                         |
#Returns:   Evaluation data frame                                                         |
#=========================================================================================|

ModelGather = function(inputModel,
                       dfTraining,
                       dfValidation,
                       dfPredicted){
    
    #Test variables
    # inputModel <- model_GLMNet
    # dfTraining <- GCD_Training
    # dfValidation <- GCD_Validation
    # dfPredicted <- GCD_Predict
    # dfx <- inputModel$pred
    
#row.names(inputModel$bestTune)
#for glmnet use alpha and lambda to choose    
    
    #(Conditionally detach MASS package to avoid conflict with dplyr SELECT)
    p_unload(MASS) 
    
    dfEval_Tr <- dfTraining %>%
        mutate(Partition="Training",
               Pred_Prob = inputModel$pred$Bad,
               Predicted = inputModel$pred$pred,
               Observed  = inputModel$pred$obs) %>%
        select(Partition,ID,Pred_Prob,Predicted,Observed)%>%
        arrange(desc(Pred_Prob))
    
    dfEval_Val <- dfValidation %>%
        mutate(Partition="Validation",
               Pred_Prob = dfPredicted$Bad,
               Predicted = factor(as.character(ifelse(dfPredicted$Bad > 0.5,"Bad","Good")),
                                  levels=c("Good", "Bad"), ordered=TRUE)) %>% 
        select(Partition,ID,Pred_Prob,Predicted,Observed=Class) %>%
        arrange(desc(Pred_Prob))
    
    #Compile the Model objects and delete the working steps
    dfEvaluation <- bind_rows(dfEval_Tr,dfEval_Val)
    rm(dfEval_Tr,dfEval_Val)
    
    return(dfEvaluation)
}



#=========================================================================================|
#FUNCTION: Model Plots                                                                    |
#                                                                                         |
#Purpose:   produces ROC and Lift plots from Model Evaluation data                        |
#Accepts:   Evaluation data frame with training and validation data                       |
#Returns:   List of Plots (ROC, Lift)                                                     |
#=========================================================================================|
ModelPlots = function(dfEvaluation){
    
    
    #Use ROCR to extract ROC and Lift plots and send to ggplot
    library(ROCR)
    library(ggplot2)
    p_unload(MASS) #(Detach MASS package if previously called by caret to avoid conflict with dplyr SELECT statements)
    
    # ROC Plots and Lift Charts ===================================================================
    
    #Generate the Plot objects for the training partition
    DF_Plot_Tr <- dfEvaluation %>%
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
    DF_Plot_Val <- dfEvaluation %>%
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
    #ROC_Plot
    
    
    #Plot Lift to GGPlot
    Lift_Plot <- ggplot(data=Evaluate_Results_DF,aes(x=Lift_X,y=Lift_Y,col=Partition)) +
        geom_line(size=1.1) +
        scale_color_manual(values=c("#0000FF", "#FF0000")) +
        geom_abline(intercept = 1, slope = 0, color='grey') +
        labs(title = 'Lift Chart', y = 'Percentile', x = 'Lift') +
        theme_minimal()
    #Lift_Plot
    
    PlotList <- list("ROC" = ROC_Plot, "Lift" = Lift_Plot)
    rm(Evaluate_Results_DF, DF_Plot_Tr, DF_Plot_Val)
    
    return(PlotList)
    
}


#=========================================================================================|
#FUNCTION: Model Variable Importance                                                      |
#                                                                                         |
#Purpose:   produces VarImp bar chart for eligible model types                            |
#Accepts:   A Model object                                                                |
#Returns:   A VarImp plot object                                                          |
#=========================================================================================|

ModelVarImp = function(inputModel){

    #Test variables
    #inputModel <- model_GLM
    
    #Variable Importance plot
    df_vi1 <- inputModel$finalModel$coefficients
    df_vi2 <- data.frame(varx=names(df_vi1), value=df_vi1, row.names=NULL) %>%
        mutate(variable=as.character(varx),
               sign=ifelse(value/abs(value)>0,"+ve","-ve")) %>%
        select(variable,sign) %>%
        filter(variable != "(Intercept)")
    df_vi <- varImp(inputModel)$importance %>%
        mutate(names=row.names(.)) %>%
        select(variable=names,ranking=Overall) %>%
        left_join(df_vi2, by = "variable") %>%
        arrange(-ranking)
    VI_Plot <- ggplot(df_vi, aes(x=reorder(variable, ranking), y=ranking, fill=sign)) +
        geom_bar(stat='identity') +
        coord_flip() +
        scale_fill_manual("legend", values = c("+ve" = "blue", "-ve" = "red"))
    rm(df_vi1,df_vi2,df_vi)
    
    return(VI_Plot)
    
}


#===============================================================================|
#FUNCTION: Model Evaluation (original)                                          |
#                                                                               |
#Purpose:   Processes caret models and returns a series of plots and charts to  |
#               evaluate the results.                                           |
#Accepts:   Processes caret models and returns a series of plots and charts to  |
#               evaluate the results.                                           |
#===============================================================================|

# ModelEvaluate = function(GCD_Evaluate,GCD_Model){
# 
# 
#     #Use ROCR to extract ROC and Lift plots and send to ggplot
#     library(ROCR)
#     library(ggplot2)
#     p_unload(MASS) #(Detach MASS package if previously called by caret to avoid
#     #conflict with dplyr SELECT statements)
#     
#     # ROC Plots and Lift Charts ===================================================================
#     
#     #Generate the Plot objects for the training partition
#     DF_Plot_Tr <- GCD_Evaluate %>%
#         filter(Partition=="Training") %>% 
#         mutate(bin = ntile(Pred_Prob, 20)) %>% 
#         arrange(desc(bin)) %>% 
#         select(Partition,ID,bin,everything())
#     
#     PredObj_Train <- prediction(predictions = DF_Plot_Tr$bin,
#                                 labels      = DF_Plot_Tr$Observed)
#     ROC_Train <- performance(PredObj_Train, measure="tpr", x.measure="fpr")
#     Lift_Train <- performance(PredObj_Train, measure="lift", x.measure="rpp")
#     AUC_Train = round(100*performance(PredObj_Train ,"auc")@y.values[[1]],1)
#     
#     #Generate the Plot objects for the validation partition
#     DF_Plot_Val <- GCD_Evaluate %>%
#         filter(Partition=="Validation") %>% 
#         mutate(bin = ntile(Pred_Prob, 20)) %>% 
#         arrange(desc(bin)) %>% 
#         select(Partition,ID,bin,everything())
#     
#     
#     PredObj_Valid <- prediction(predictions = DF_Plot_Val$bin,
#                                 labels      = DF_Plot_Val$Observed)
#     ROC_Valid <- performance(PredObj_Valid, measure="tpr", x.measure="fpr")
#     Lift_Valid <- performance(PredObj_Valid, measure="lift", x.measure="rpp")
#     AUC_Valid = round(100*performance(PredObj_Valid ,"auc")@y.values[[1]],1)
#     
#     
#     #Combine plot evaluation results into a single data frame
#     Evaluate_Results_DF <- bind_rows(
#         data.frame(Partition="Training",
#                    ROC_X=unlist(ROC_Train@x.values),
#                    Lift_X=unlist(Lift_Train@x.values),
#                    ROC_Y=unlist(ROC_Train@y.values),
#                    Lift_Y=unlist(Lift_Train@y.values))
#         ,
#         data.frame(Partition="Validation",
#                    ROC_X=unlist(ROC_Valid@x.values),
#                    Lift_X=unlist(Lift_Valid@x.values),
#                    ROC_Y=unlist(ROC_Valid@y.values),
#                    Lift_Y=unlist(Lift_Valid@y.values)) 
#     )
# 
#     
#     #Plot ROC to GGPlot
#     ROC_Plot <- ggplot(data=Evaluate_Results_DF,aes(x=ROC_X,y=ROC_Y,col=Partition)) +
#         geom_line(size=1.1) +
#         scale_color_manual(values=c("#0000FF", "#FF0000")) +
#         geom_abline(intercept = 0, slope = 1, color='grey') +
#         annotate("text", x = .742, y = .25,
#                  label = paste0('AUC (Training): ', AUC_Train, '%'), size = 4) +
#         annotate("text", x = .75, y = .18,
#                  label = paste0('AUC (Validation): ', AUC_Valid, '%'), size = 4) +
#         labs(title = 'ROC Chart', y = 'Sensitivity (%)', x = 'Specificity (%)') +
#         theme_minimal()
#     #ROC_Plot
#     
#     
#     #Plot Lift to GGPlot
#     Lift_Plot <- ggplot(data=Evaluate_Results_DF,aes(x=Lift_X,y=Lift_Y,col=Partition)) +
#         geom_line(size=1.1) +
#         scale_color_manual(values=c("#0000FF", "#FF0000")) +
#         geom_abline(intercept = 1, slope = 0, color='grey') +
#         labs(title = 'Lift Chart', y = 'Percentile', x = 'Lift') +
#         theme_minimal()
#     #Lift_Plot
#    
#     PlotList <- list("ROC" = ROC_Plot, "Lift" = Lift_Plot)
#     
# 
#     
#     #Variable Importance plot
#     df_vi1 <- GCD_Model$finalModel$coefficients
#     df_vi2 <- data.frame(varx=names(df_vi1), value=df_vi1, row.names=NULL) %>%
#         mutate(variable=as.character(varx),
#                sign=ifelse(value/abs(value)>0,"+ve","-ve")) %>%
#         select(variable,sign) %>%
#         filter(variable != "(Intercept)")
#     df_vi <- varImp(GCD_Model)$importance %>%
#         mutate(names=row.names(.)) %>%
#         select(variable=names,ranking=Overall) %>%
#         left_join(df_vi2, by = "variable") %>%
#         arrange(-ranking)
#     VI_Plot <- ggplot(df_vi, aes(x=reorder(variable, ranking), y=ranking, fill=sign)) +
#         geom_bar(stat='identity') +
#         coord_flip() +
#         scale_fill_manual("legend", values = c("+ve" = "blue", "-ve" = "red"))
#     rm(df_vi1,df_vi2,df_vi)
#     #VI_Plot
#     
#     
#     
#     # Confusion Matrix =========================================================================
#     CM <- confusionMatrix(data = DF_Plot_Val$Predicted,
#                           reference = DF_Plot_Val$Observed)
#     CM
#     
#     
#     
#     # Gains tables =============================================================================
#     DF_Gains_Val <- GCD_Evaluate %>%
#         filter(Partition=="Validation") %>%
#         mutate(bin = ntile(1-Pred_Prob, 10)) %>% 
#         group_by(bin) %>%
#         summarise(
#             observations=n(),
#             bads = sum(Observed=="Bad")
#         ) %>% 
#         mutate(cum.observations = cumsum(observations),
#                cum.bads = cumsum(bads),
#                gain=cum.bads/sum(bads)*100,
#                cum.lift=gain/(bin*(100/10))) %>%
#         ungroup() %>% 
#         select(decile=bin,observations,cum.observations,bads,cum.bads,gain,cum.lift)
#     
#     DF_Gains_Tr <- GCD_Evaluate %>%
#         filter(Partition=="Training") %>%
#         mutate(bin = ntile(1-Pred_Prob, 10)) %>% 
#         group_by(bin) %>%
#         summarise(
#             observations=n(),
#             bads = sum(Observed=="Bad")
#         ) %>% 
#         mutate(cum.observations = cumsum(observations),
#                cum.bads = cumsum(bads),
#                gain=cum.bads/sum(bads)*100,
#                cum.lift=gain/(bin*(100/10))) %>%
#         ungroup() %>% 
#         select(decile=bin,observations,cum.observations,bads,cum.bads,gain,cum.lift)
#     
#     
#     
#     #Gather object elements into a list:
#     PlotList <- list("ROC" = ROC_Plot, "Lift" = Lift_Plot, "VarImp" = VI_Plot)
#     TableList <- list("Gains_Tr" = DF_Gains_Tr, "Gains_Val" = DF_Gains_Val, "CM" = CM)
#     EvaluationList <- list("Plots" = PlotList,
#                            "Tables" = TableList)
# 
#     return(EvaluationList)
# 
#     #Remove interim objects
#     rm(AUC_Train, AUC_Valid, Lift_Train, Lift_Valid, ROC_Train, ROC_Valid,
#         PredObj_Train, PredObj_Valid, CM, Evaluate_Results_DF,
#         DF_Plot_Tr, DF_Plot_Val, DF_Gains_Tr, DF_Gains_Val,
#         PlotList, TableList, EvaluationList, ROC_Plot, Lift_Plot)
# 
# }
