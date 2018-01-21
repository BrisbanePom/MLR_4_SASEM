#=========================================================================================|
#FUNCTION: Model Gather                                                                   |
#                                                                                         |
#Purpose:   Gather the validation and training data into the evaluation data frame        |
#Accepts:   Training data, validation data, model                                         |
#Returns:   Evaluation data frame                                                         |
#=========================================================================================|

ModelGather = function(model_Input,
                       df_Input_Train,
                       df_Input_HoldOut)
    {

    #(Conditionally detach MASS package to avoid conflict with dplyr SELECT)
    p_unload(MASS)
    
    #Gather and clean the training data  
    df_Predict_Tr <- predict(model_Input, df_Input_Train, type="prob")
    df_Eval_Tr <- df_Input_Train %>%
                    mutate(Partition = "Training",
                           Method = model_Input$method,
                           Pred_Prob = df_Predict_Tr$Bad,
                           Predicted = predict(model_Input, df_Input_Train, type="raw"),
                           Observed  = Class) %>%
                    select(Method,Partition,ID,Pred_Prob,Predicted,Observed) %>%
                    arrange(desc(Pred_Prob))
    
    
    #Gather and clean the HoldOut data
    df_Predict_HO <- predict(model_Input, df_Input_HoldOut, type="prob")
    df_Eval_HO <- df_Input_HoldOut %>%
        mutate(Partition = "Hold Out",
               Method = model_Input$method,
               Pred_Prob = df_Predict_HO$Bad,
               Predicted = predict(model_Input, df_Input_HoldOut, type="raw"),
               Observed  = Class) %>%
        select(Method,Partition,ID,Pred_Prob,Predicted,Observed) %>%
        arrange(desc(Pred_Prob))
    
    #Compile the Model objects and delete the working steps
    df_Evaluation <- bind_rows(df_Eval_Tr,df_Eval_HO)
    rm(df_Predict_Tr,df_Predict_HO,df_Eval_Tr,df_Eval_HO)
    
    return(df_Evaluation)
}



#=========================================================================================|
#FUNCTION: Model Plots                                                                    |
#                                                                                         |
#Purpose:   produces ROC and Lift plots from Model Evaluation data                        |
#Accepts:   Evaluation data frame with training and validation data                       |
#Returns:   List of Plots (ROC, Lift)                                                     |
#=========================================================================================|
ModelPlots = function(df_Evaluation){
    
    
    #Use ROCR to extract ROC and Lift plots and send to ggplot
    library(ROCR)
    library(ggplot2)
    p_unload(MASS) #(Detach MASS package if previously called by caret to avoid conflict with dplyr SELECT statements)
    
    # ROC Plots and Lift Charts ===================================================================
    
    #Generate the Plot objects for the training partition
    DF_Plot_Tr <- df_Evaluation %>%
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
    DF_Plot_Val <- df_Evaluation %>%
        filter(Partition=="Hold Out") %>% 
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
        data.frame(Partition="Hold Out",
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
                 label = paste0('AUC (Hold Out): ', AUC_Valid, '%'), size = 4) +
        labs(title = 'ROC Chart', y = 'Sensitivity (%)', x = '1 - Specificity (%)') +
        theme_minimal()
    #ROC_Plot
    
    
    #Plot Lift to GGPlot
    Lift_Plot <- ggplot(data=Evaluate_Results_DF,aes(x=Lift_X,y=Lift_Y,col=Partition)) +
        geom_line(size=1.1) +
        scale_color_manual(values=c("#0000FF", "#FF0000")) +
        geom_abline(intercept = 1, slope = 0, color='grey') +
        labs(title = 'Lift Chart', y = 'Cumulative Lift', x = 'Percentile') +
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
