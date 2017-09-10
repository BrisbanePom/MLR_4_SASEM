
#Cumulative gains table - validation data
DF_Gains_Val <- GCD_Evaluate %>%
    filter(Partition=="Validation") %>%
    mutate(bin = ntile(1-Pred_Prob, 10)) %>% 
    group_by(bin) %>%
    summarise(
        observations=n(),
        bads = sum(Observed=="Bad")
    ) %>% 
    mutate(cum.observations = cumsum(observations),
           cum.bads = cumsum(bads),
           gain=cum.bads/sum(bads)*100,
           cum.lift=gain/(bin*(100/10))) %>%
    ungroup() %>% 
    select(decile=bin,observations,cum.observations,bads,cum.bads,gain,cum.lift)

DF_Gains_Tr <- GCD_Evaluate %>%
    filter(Partition=="Training") %>%
    mutate(bin = ntile(1-Pred_Prob, 10)) %>% 
    group_by(bin) %>%
    summarise(
        observations=n(),
        bads = sum(Observed=="Bad")
    ) %>% 
    mutate(cum.observations = cumsum(observations),
           cum.bads = cumsum(bads),
           gain=cum.bads/sum(bads)*100,
           cum.lift=gain/(bin*(100/10))) %>%
    ungroup() %>% 
    select(decile=bin,observations,cum.observations,bads,cum.bads,gain,cum.lift)


#Print to markdown via knitr


mutate(Sum_Target = count(Predicted=="Bad")) 


%>% #,
    #Cum_Sum_Target = cumsum(Sum_Target)) %>%
    
    arrange(desc(bin)) %>%
    select(Partition,ID,bin,everything())


# %>% 
#     arrange(desc(Pred_Prob))
# 
# 
# 
# group_by(GainBin) %>%
#     mutate(Sum_Target = sum(Target),
#            Cum_Sum_Target = cumsum(Sum_Target)) %>%
#     ungroup() %>% 
#     %>% 
#     arrange(desc(Pred_Prob))
# arrange(desc(Pred_Prob)) %>% 

#VARIMP
obj1 <- GCD_Model_LR$finalModel$coefficients

col_index <- varImp(GCD_Model_LR)$importance %>% 
    mutate(names=row.names(.)) %>%
    arrange(-Overall)






