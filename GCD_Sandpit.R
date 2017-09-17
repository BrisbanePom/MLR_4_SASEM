#Print results to markdown via knitr



#Variable Importance - including signs
df_vi1 <- GCD_Model_LR$finalModel$coefficients
df_vi2 <- data.frame(varx=names(df_vi1), value=df_vi1, row.names=NULL) %>%
        mutate(variable=as.character(varx),
               sign=ifelse(value/abs(value)>0,"+ve","-ve")) %>%
        select(variable,sign) %>% 
        filter(variable != "(Intercept)")
df_vi <- varImp(GCD_Model_LR)$importance %>% 
    mutate(names=row.names(.)) %>%
    select(variable=names,ranking=Overall) %>% 
    left_join(df_vi2, by = "variable") %>% 
    arrange(-ranking)
ggplot(df_vi, aes(x=reorder(variable, ranking), y=ranking, fill=sign)) +
    geom_bar(stat='identity') +
    coord_flip() + 
    scale_fill_manual("legend", values = c("+ve" = "blue", "-ve" = "red"))
rm(df_vi1,df_vi2,df_vi)


