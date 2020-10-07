library(tidyverse)
library(brglm)
library(magrittr)


#################
# Functions
#################

model_f_dm <- function(dt, arm){
  brglm(f_dm1, data = dt %>% filter(action == arm))
}


get_pred_reward <- function(k, arm){
  pred_action_reward %>%
    filter(id == k) %>% select(paste0('r', arm)) %>%
    as.numeric()
}


lookup_data <- function(k){
  data %>% filter(id == k) %>% select(id, action, reward, prop)
}


epsilon_greedy <- function(eps, tally){
  rand <- runif(1)
  if (rand < eps) {
    result <- tibble(arm = sample(arms, 1), prob = eps*0.1)
  } else {
    best_arm <- tally %>% slice_max(conv_prob, with_ties = FALSE) %>% .$action
    if (identical(best_arm, integer(0))){
      result <- tibble(arm = sample(arms, 1), prob = eps*0.1)
    } else {
      result <- tibble(arm = best_arm, prob = (1 - eps) * 1)
    }
  }
  return(result)
}



#################
# Load data
#################

url <- 'http://d1ie9wlkzugsxr.cloudfront.net/data_cmab_basic/dataset.txt'

names <- c('action', 'reward', str_c('x', seq(1:100)), 'empty_col')

raw_data <- read_delim(url, delim = ' ', col_names = names) %>%
  select(-empty_col) %>%
  mutate(id = row_number()) %>%
  select(id, action, reward, starts_with('x'))


# Action propensities
propensities <- raw_data %>%
  group_by(action) %>%
  summarise(n = n(), .groups = 'drop') %>%
  mutate(prop = n / sum(n)) %>%
  select(-n)



data <- raw_data %>% left_join(propensities, by = 'action')

# Actual conversion rate
sum(data$reward) / nrow(data)
# [1] 0.1039

# Action pool
arms <- sort(unique(data$action))


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

names(pred_action_reward_list) <- map_chr(arms, ~paste0('r', .))

map(pred_action_reward_list, length)

pred_action_reward <- pred_action_reward_list %>%
  bind_cols(.id = data$id) %>%
  select(id = .id, starts_with('r'))


#######################################
# Doubly robust non-stationary algorithm
#######################################

# Fixed parameters
eps <- 0.5
c_max <- 1
q <- 0.01




# Initialise values
ht <- tibble(id = integer(), action = integer(), reward = numeric())
t <- 0
ct <- c_max
C <- 0
V_dr <- 0
Q <- tibble(Q = numeric())
set.seed(666)

for (k in 1:200){
  # Get action
  conv_probs <- ht %>%
    select(id, action, reward) %>%
    group_by(action) %>%
    summarise(total = n(), conversions = sum(reward), .groups = 'drop') %>%
    mutate(conv_prob = conversions / total)

  pi_result <- epsilon_greedy(eps, conv_probs)

  # (1)
  pred_reward_proposed <- get_pred_reward(k, pi_result$arm)
  pred_reward_actual <- get_pred_reward(k, lookup_data(k)$action)
  reward_diff <- lookup_data(k)$reward - pred_reward_actual
  Vk <- pred_reward_proposed + (pi_result$prob / lookup_data(k)$prop) * reward_diff

  # (2)
  V_dr <- V_dr + ct * Vk

  # (3)
  C <- C + ct

  # (4)
  Q %<>% bind_rows(tibble(Q = lookup_data(k)$prop / pi_result$prob))

  # (5)
  uk <- runif(1)

  # (6)
  if ( uk <= ct * pi_result$prob / lookup_data(k)$prop ){
    ht %<>% bind_rows(tibble(id = lookup_data(k)$id,
                             action = lookup_data(k)$action,
                             reward = lookup_data(k)$reward))
    ct <- c(c_max, quantile(Q$Q, probs = q) %>% as.numeric()) %>% min()
    t <- t + 1
  }

  print(paste0('k:', k, ', t:', t, ', Cumulative:', signif(V_dr, 3), ', Average:', signif(V_dr / C, 3), ', Vk:', signif(Vk, 3)))

}

