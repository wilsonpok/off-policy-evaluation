library(tidyverse)
library(ggthemes)
library(brglm)


#################
# Functions
#################

get_summary_results <- function(data){
  data %>%
    group_by(action) %>%
    summarise(n = n(),
              sum_reward = sum(reward),
              mean_reward = mean(reward),
              error_min = qbeta(0.025, 1 + sum_reward, 1 + n - sum_reward),
              error_max = qbeta(0.975, 1 + sum_reward, 1 + n - sum_reward),
              .groups = 'drop') %>%
    mutate(action = action %>% as.factor())
}


plot_summary_results <- function(data){
  data %>%
    ggplot(aes(x = action, y = mean_reward)) +
    geom_point() +
    geom_errorbar(aes(ymin = error_min, ymax = error_max), width = 0.1) +
    theme_fivethirtyeight() +
    theme(axis.title = element_text()) +
    scale_x_discrete(name = 'Action') +
    scale_y_continuous(name = 'Conversion rate', labels = scales::percent)
}


get_summary_table <- function(data){
  data %>%
    summarise(n = n(), .groups = 'drop') %>%
    mutate(prop = n / sum(n))
}


get_beta_dist <- function(arm){
  tot <- counts %>% filter(action == arm) %>% .$total
  conv <- counts %>% filter(action == arm) %>% .$conversions
  tibble(action = arm, x = xrange, y = dbeta(xrange, 1 + conv, 1 + tot - conv))
}


model_f_dm <- function(dt, arm){
  brglm(f_dm1, data = dt %>% filter(action == arm))
}


pred_reward_actual_action <- function(arm){
  tibble(id = filter(data, action == arm)$id,
         pred_reward_for_actual_action = predict(models_list[[arm]],
                                                 filter(data, action == arm),
                                                 type = 'response'))
}


get_policy_beta <- function(pol, num_conv){
  tibble(policy = pol,
         x = xrange,
         y = dbeta(xrange,
                   1 + num_conv,
                   1 + nrow(data) - num_conv))
}


#################
# Load data
#################

url <- 'http://d1ie9wlkzugsxr.cloudfront.net/data_cmab_basic/dataset.txt'

names <- c('action', 'reward', str_c('x', seq(1:100)), 'empty_col')

data <- read_delim(url, delim = ' ', col_names = names) %>%
  select(-empty_col) %>%
  mutate(id = row_number()) %>%
  select(id, action, reward, starts_with('x'))


#################
# Summarise data
#################

data %>%
  group_by(action) %>%
  get_summary_results() %>%
  plot_summary_results()

data %>%
  group_by(action) %>%
  get_summary_table()

data %>%
  group_by(reward) %>%
  get_summary_table()

# Actual conversion rate
sum(data$reward) / nrow(data)
# [1] 0.1039

# Action propensities
propensities <- data %>%
  group_by(action) %>%
  summarise(n = n(), .groups = 'drop') %>%
  mutate(prop = n / sum(n))

# Action pool
arms <- sort(unique(data$action))


###################################################
# Beta distributions per action for original policy
###################################################

xrange <- seq(0, 0.3, 1e-3)

counts <- data %>%
  group_by(action) %>%
  summarise(total = n(), conversions = sum(reward), .groups = 'drop')

beta_dist_by_action <- map_dfr(arms, get_beta_dist)

beta_dist_by_action %>%
  mutate(action = action %>% as.factor()) %>%
  ggplot(aes(x = x, y = y, colour = action)) +
  geom_line(size = 1) +
  facet_wrap(~ action, nrow = 5) +
  theme_fivethirtyeight() +
  theme(legend.position = 'none')




############################
# Define a proposed policy
############################

# Choose randomly from actions 1-5 only
set.seed(666)
proposed_policy <- tibble(id = data$id,
                          proposed_action = sample(seq(1, 5),
                                                   nrow(data),
                                                   replace = TRUE))


############################
# Fit a model to each action
############################

# Specify formula
context_cols <- data %>% select(starts_with('x')) %>% names()
f_dm1 <- as.formula(paste('reward ~', paste(context_cols, collapse = '+')))

# Train models
models_list <- map(.f = model_f_dm, .x = arms, dt = data)


##################################
# Predict reward for every action
##################################

pred_action_reward_list <- map(models_list, .f = ~predict(., data,
                                                          type = 'response'))

names(pred_action_reward_list) <- map_chr(seq(1, 10), ~paste0('r', .))

map(pred_action_reward_list, length)

pred_action_reward <- pred_action_reward_list %>%
  bind_cols(.id = data$id) %>%
  select(id = .id, starts_with('r'))


##################################
# Predict reward for actual action
##################################

pred_actual_reward <- map_dfr(arms, pred_reward_actual_action) %>%
  arrange(id)


#################
# Direct method
#################

dm_rewards <- proposed_policy %>%
  inner_join(pred_action_reward, by = 'id') %>%
  mutate(pred_reward_for_proposed_action = case_when(proposed_action == 1 ~ r1,
                                 proposed_action == 2 ~ r2,
                                 proposed_action == 3 ~ r3,
                                 proposed_action == 4 ~ r4,
                                 proposed_action == 5 ~ r5,
                                 proposed_action == 6 ~ r6,
                                 proposed_action == 7 ~ r7,
                                 proposed_action == 8 ~ r8,
                                 proposed_action == 9 ~ r9,
                                 proposed_action == 10 ~ r10))

# DM conversion rate
sum(dm_rewards$pred_reward_for_proposed_action) / nrow(data)
# 0.1366361


##################################
# Inverse propensity scores
##################################

ips_rewards <- data %>%
  select(id, actual_action = action, actual_reward = reward) %>%
  inner_join(proposed_policy, by = 'id') %>%
  mutate(propensity = case_when(proposed_action == 1 ~ propensities$prop[[1]],
                                proposed_action == 2 ~ propensities$prop[[2]],
                                proposed_action == 3 ~ propensities$prop[[3]],
                                proposed_action == 4 ~ propensities$prop[[4]],
                                proposed_action == 5 ~ propensities$prop[[5]],
                                proposed_action == 6 ~ propensities$prop[[6]],
                                proposed_action == 7 ~ propensities$prop[[7]],
                                proposed_action == 8 ~ propensities$prop[[8]],
                                proposed_action == 9 ~ propensities$prop[[9]],
                                proposed_action == 10 ~ propensities$prop[[10]])) %>%
  mutate(weighted_reward = ifelse(actual_action == proposed_action,
                                  actual_reward / propensity, 0))

# IPS conversion rate
sum(ips_rewards$weighted_reward) / nrow(data)
# [1] 0.1048184


#################
# Doubly robust
#################

dr_rewards <- dm_rewards %>%
  select(id, proposed_action, pred_reward_for_proposed_action) %>%
  inner_join(ips_rewards %>% select(-weighted_reward),
             by = c('id', 'proposed_action')) %>%
  inner_join(pred_actual_reward, by = 'id') %>%
  mutate(reward_diff = actual_reward - pred_reward_for_actual_action) %>%
  mutate(weighted_reward_diff = ifelse(actual_action == proposed_action,
                                       reward_diff / propensity, 0)) %>%
  mutate(dr_result = pred_reward_for_proposed_action + weighted_reward_diff)

# DR conversion rate
sum(dr_rewards$dr_result) / nrow(data)
# [1] 0.1033649



#############################################################
# ## Compare conversion rates of actual and proposed policies
#############################################################

beta_actual <- get_policy_beta('Actual',
                               num_conv = sum(data$reward))

beta_proposed_dm <- get_policy_beta('Proposed (dm)',
                                    num_conv = sum(dm_rewards$pred_reward_for_proposed_action))

beta_proposed_ips <- get_policy_beta('Proposed (ips)',
                                    num_conv = sum(ips_rewards$weighted_reward))

beta_proposed_dr <- get_policy_beta('Proposed (dr)',
                                     num_conv = sum(dr_rewards$dr_result))


all_betas <- beta_actual %>%
  bind_rows(beta_proposed_dm) %>%
  bind_rows(beta_proposed_ips) %>%
  bind_rows(beta_proposed_dr)


all_betas %>%
  filter(policy %in% c('Actual', 'Proposed (dr)')) %>%
  ggplot(aes(x = x, y = y, colour = policy)) +
  geom_line(size = 1) +
  scale_colour_few() +
  theme_fivethirtyeight()
