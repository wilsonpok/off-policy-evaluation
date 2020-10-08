library(tidyverse)
library(magrittr)
library(ggthemes)


#################
# Functions
#################

get_conv_by_action <- function(dt){
  dt %>%
    group_by(action) %>%
    summarise(total = n(), conversions = sum(reward), .groups = 'drop') %>%
    mutate(conv_prob = conversions / total)
}


get_pred_reward <- function(arm){
  pred_rewards %>% filter(action == arm) %>% .$conv_prob
}


lookup_data <- function(k){
  data %>% filter(id == k) %>% select(id, action, reward, prop)
}


epsilon_greedy <- function(eps, tally){
  rand <- runif(1)
  if (rand < eps) {
    result <- tibble(ex = 'explore', arm = sample(arms, 1), prob = eps * 1/length(arms))
  } else {
    best_arm <- tally %>% slice_max(conv_prob, with_ties = FALSE) %>% .$action
    if (identical(best_arm, integer(0))){
      result <- tibble(ex = 'pseudo-explore', arm = sample(arms, 1), prob = eps * 1/length(arms))
    } else {
      result <- tibble(ex = 'exploit', arm = best_arm, prob = (1 - eps) * 1)
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
(propensities <- raw_data %>%
  group_by(action) %>%
  summarise(n = n(), .groups = 'drop') %>%
  mutate(prop = n / sum(n)) %>%
  select(-n))

# action   prop
# <dbl>  <dbl>
# 1      1 0.102
# 2      2 0.0982
# 3      3 0.0974
# 4      4 0.105
# 5      5 0.100
# 6      6 0.0963
# 7      7 0.104
# 8      8 0.0999
# 9      9 0.0988
# 10     10 0.0987


data <- raw_data %>% left_join(propensities, by = 'action')


# Predicted rewards
(pred_rewards <- raw_data %>% get_conv_by_action())

# action total conversions conv_prob
# <dbl> <int>       <dbl>     <dbl>
# 1      1  1020          21    0.0206
# 2      2   982         263    0.268
# 3      3   974         138    0.142
# 4      4  1047          54    0.0516
# 5      5  1005          54    0.0537
# 6      6   963          93    0.0966
# 7      7  1035         201    0.194
# 8      8   999          28    0.0280
# 9      9   988         157    0.159
# 10     10   987          30    0.0304

# Actual conversion rate
sum(data$reward) / nrow(data)
# [1] 0.1039

# Action pool
arms <- sort(unique(data$action))

# Length of historical data to use
k_length <- nrow(data) / 5 # Just so loop ends quickly




#######################################
# Doubly robust non-stationary algorithm
#######################################

# Fixed parameters
eps <- 0.5
c_max <- 1
q <- 0.01


# Initialise values
ht <- tibble(t = integer(),
             ex = character(),
             id = integer(),
             action = integer(),
             reward = numeric(),
             new_action = integer())
t <- 0
ct <- c_max
C <- 0
V_dr <- 0
Q <- tibble(Q = numeric())



# Iterate over historical data
for (k in 1:k_length){
  # Get action
  conv_probs <- ht %>% get_conv_by_action()

  pi_result <- epsilon_greedy(eps, conv_probs)

  # (1)
  pred_reward_proposed <- get_pred_reward(pi_result$arm)

  pred_reward_actual <- get_pred_reward(lookup_data(k)$action)

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
    ht %<>% bind_rows(tibble(t = t,
                             ex = pi_result$ex,
                             id = lookup_data(k)$id,
                             action = lookup_data(k)$action,
                             reward = lookup_data(k)$reward,
                             new_action = pi_result$arm))
    ct <- c(c_max, quantile(Q$Q, probs = q) %>% as.numeric()) %>% min()
    t <- t + 1
  }

  print(paste0('k:', k, ', t:', t, ', Cumulative:', signif(V_dr, 3), ', Average:', signif(V_dr / C, 3)))

}


#################
# Results
#################

conv_probs

ht %>%
  mutate(new_action = new_action %>% as.factor()) %>%
  ggplot(aes(x = t, y = ex, colour = new_action)) +
  geom_point() +
  theme_fivethirtyeight()

ht %>%
  mutate(row_id = row_number()) %>%
  mutate(row_group = row_id %>% cut_width(width = k_length/50)) %>%
  mutate(new_action = new_action %>% as.factor()) %>%
  ggplot(aes(x = row_group, fill = new_action)) +
  geom_bar() +
  theme_fivethirtyeight() +
  theme(axis.text.x = element_blank())



####################################
# Sample historical data
####################################

samp <- data %>% slice_sample(n = t)

samp %>%
  select(id, action, reward) %>%
  group_by(action) %>%
  summarise(total = n(), conversions = sum(reward), .groups = 'drop') %>%
  mutate(conv_prob = conversions / total)


#######################################
# Compare historical to epsilon-greedy
#######################################

# No bootstrap, just a single run of each

reward_epsilon_greedy <- V_dr
total_epsilon_greedy <- C

reward_random_sample <- samp$reward %>% sum()
total_random_sample <- t


xrange <- seq(0, 0.3, length.out = 200)

beta_epsilon_greedy <- tibble(policy = 'epsilon-greedy',
                              x = xrange,
                              y = dbeta(x,
                                        1 + reward_epsilon_greedy,
                                        1 + total_epsilon_greedy - reward_epsilon_greedy))

beta_random_sample <- tibble(policy = 'random-sample',
                              x = xrange,
                              y = dbeta(x,
                                        1 + reward_random_sample,
                                        1 + total_random_sample - reward_random_sample))


beta_epsilon_greedy %>%
  bind_rows(beta_random_sample) %>%
  ggplot(aes(x = x, y = y, colour = policy)) +
  geom_line(size = 1) +
  scale_colour_fivethirtyeight() +
  theme_fivethirtyeight()

