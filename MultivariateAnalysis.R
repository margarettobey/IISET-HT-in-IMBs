library(tidyverse)
library(broom)
library(gtsummary)
library(stats)

data <- read.csv("data_categorical_9070and403.csv")
#print(data)

# pacman::p_load(
#   rio,          # File import
#   here,         # File locator
#   tidyverse,    # data management + ggplot2 graphics, 
#   stringr,      # manipulate text strings 
#   purrr,        # loop over objects in a tidy way
#   broom,        # tidy up results from regressions
#   lmtest,       # likelihood-ratio tests
#   parameters,   # alternative to tidy up results from regressions
#   see          # alternative to visualise forest plots
# )


## convert dichotomous variables to 0/1 
binary_vars <- c("inRubmaps", "inSG", "has_subpremise", "yelp_massageCat", 
                 "yelp_spaCat", "yelp_category_reflexology", "yelp_reviewRating_min_NEW_is5", 
                 "yelp_average_all_ratings_NEW_moreThan4", "yelp_phone_advertisement",
                 "census_nonfamily_and_20s_both_high","census_nonfamily_and_20s_both_low",
                 "RUCA_category_metro","yelp_business_name_foot","yelp_business_name_happy",
                 "yelp_business_name_asian","yelp_business_name_touch","yelp_business_name_NEW_Combine")

data <- data %>%  
  mutate(across(                                      
    .cols = all_of(c(binary_vars, "label")),  ## for each column listed and "outcome"
    .fns = ~case_when(                              
      . %in% c(1, 1.0) ~ 1,           ## recode male, yes and death to 1
      . %in% c(0, 0.0) ~ 0,           ## female, no and recover to 0
      TRUE                                              ~ NA_real_)    ## otherwise set to missing
  )
  )

data$yelp_reviewRating_std_NEW = fct_relevel(data$yelp_reviewRating_std_NEW, "medium_std", after = 0)
data$yelp_revCount_NEW = fct_relevel(data$yelp_revCount_NEW, "5to20", after = 0)
data$yelp_lexicon_score_mean_NEW = fct_relevel(data$yelp_lexicon_score_mean_NEW, "medium_lexiconmean", after = 0)
data$yelp_authorGender_PctMale_NEW = fct_relevel(data$yelp_authorGender_PctMale_NEW, "medium_pctmale", after = 0)
data$census_pct_nonwhite_NEW = fct_relevel(data$census_pct_nonwhite_NEW, "medium", after = 0)
data$census_pct_foreign_born_NEW = fct_relevel(data$census_pct_foreign_born_NEW, "medium", after = 0)
data$census_median_income_NEW = fct_relevel(data$census_median_income_NEW, "medium", after = 0)
data$census_pct_housing_vacant_NEW = fct_relevel(data$census_pct_housing_vacant_NEW, "medium", after = 0)
data$census_pct_of_occupied_housing_rented_NEW = fct_relevel(data$census_pct_of_occupied_housing_rented_NEW, "medium", after = 0)
data$census_pct_households_with_children_NEW = fct_relevel(data$census_pct_households_with_children_NEW, "medium", after = 0)
data$census_pct_over25_with_bachelors_NEW = fct_relevel(data$census_pct_over25_with_bachelors_NEW, "medium", after = 0)
data$census_pct_nonfamily_households_NEW = fct_relevel(data$census_pct_nonfamily_households_NEW, "medium", after = 0)
data$census_pct_20_to_29_NEW = fct_relevel(data$census_pct_20_to_29_NEW, "medium", after = 0)
data$census_pct_nonfamily_households_under25_NEW = fct_relevel(data$census_pct_nonfamily_households_under25_NEW, "medium", after = 0)
data$census_pct_nonfamily_households_15to34_NEW = fct_relevel(data$census_pct_nonfamily_households_15to34_NEW, "medium", after = 0)
data$census_pct_nonfamily_households_25to34_NEW = fct_relevel(data$census_pct_nonfamily_households_25to34_NEW, "medium", after = 0)
data$min_dist_base_NEW = fct_relevel(data$min_dist_base_NEW, "medium", after = 0)
data$min_dist_truckstop_NEW = fct_relevel(data$min_dist_truckstop_NEW, "medium", after = 0)


# Multivariate Analysis
# Run from 17 to 25; 31,32,33
# variable_lists <-c(colnames(data)[c(7:16)], colnames(data)[c(35)], colnames(data)[c(17)])
# variable_lists <-c(colnames(data)[c(7:16)], colnames(data)[c(35)], colnames(data)[c(18)])
# variable_lists <-c(colnames(data)[c(7:16)], colnames(data)[c(35)], colnames(data)[c(19)])
# variable_lists <-c(colnames(data)[c(7:16)], colnames(data)[c(35)], colnames(data)[c(20)])
# variable_lists <-c(colnames(data)[c(7:16)], colnames(data)[c(35)], colnames(data)[c(21)])
# variable_lists <-c(colnames(data)[c(7:16)], colnames(data)[c(35)], colnames(data)[c(22)])
# variable_lists <-c(colnames(data)[c(7:16)], colnames(data)[c(35)], colnames(data)[c(23)])
# variable_lists <-c(colnames(data)[c(7:16)], colnames(data)[c(35)], colnames(data)[c(24)])
# variable_lists <-c(colnames(data)[c(7:16)], colnames(data)[c(35)], colnames(data)[c(25)])
# variable_lists <-c(colnames(data)[c(7:16)], colnames(data)[c(35)], colnames(data)[c(31)])
# variable_lists <-c(colnames(data)[c(7:16)], colnames(data)[c(35)], colnames(data)[c(32)])
# variable_lists <-c(colnames(data)[c(7:16)], colnames(data)[c(35)], colnames(data)[c(33)])

mv_reg1 <- variable_lists %>%  ## begin with vector of explanatory column names
  str_c(collapse = "+") %>%     ## combine all names of the variables of interest separated by a plus
  str_c("label ~ ", .) %>%    ## combine the names of variables of interest with outcome in formula style
  glm(family = "binomial",      ## define type of glm as logistic,
      data = data)          ## define your dataset

mv_tab1 <- mv_reg1 %>%
  broom::tidy(exponentiate = TRUE, conf.int = TRUE) %>%  ## get a tidy dataframe of estimates
  mutate(across(where(is.numeric), round, digits = 3))
mv_tab1





