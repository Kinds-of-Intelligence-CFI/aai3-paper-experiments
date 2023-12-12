### Graph Plotting and Analysis for Animal-AI 3 Paper
### Author: K. Voudouris, 2023 (C)
### Version: 4.3.1 (Beagle Scouts)

### Packages and Preamble


library(FSA)
library(jsonlite)
library(lme4)
library(tidyverse)

options(scipen = 999)
set.seed(2023)

# Set working directory to path to this file

################################################################################
### Foraging Tree                                                           ####
################################################################################

dreamer <- stream_in(file("../../dreamer/logdir/foraging/foraging-eval/metrics.jsonl")) %>%
  select(`episode/score`) %>%
  mutate(`episode/score` = str_remove_all(`episode/score`, "\\[|\\]"),
         `episode/score` = as.numeric(`episode/score`)) %>%
  drop_na() %>%
  slice_sample(n = 100)

heuristic <- read.csv("../data/foraging/heuristicAgent100Foraging.csv") %>%
  select(FinalReward) %>%
  mutate(FinalReward = str_remove_all(FinalReward, "\\[|\\]"),
         FinalReward = as.numeric(FinalReward))

randomactionagent <- read.csv("../data/foraging/randomActionAgent100Foraging.csv") %>%
  select(FinalReward)

ppo <- read.csv("../data/foraging/PPORaycast100Foraging.csv") %>%
  select(FinalReward)

foraging_combined_data <- tibble(`Random Action Agent` = randomactionagent$FinalReward,
                        `Heuristic Agent` = heuristic$FinalReward,
                        `PPO (Raycast)` = ppo$FinalReward,
                        `Dreamer-v3 (64x64 Image)` = dreamer$`episode/score`) %>%
  pivot_longer(cols = everything(),
               names_to = "Agent",
               values_to = "Score") %>%
  mutate(Agent = as_factor(Agent))

sink("../data/foraging/foragingTaskOmnibusTests.txt")

cat("Foraging Task: 100 runs\n")
cat("Today's Date:", Sys.Date(), "\n")
kruskal.test(Score ~ Agent, data = foraging_combined_data)

dunnTest(Score ~ Agent, data = foraging_combined_data,
         method="bonferroni")

sink()

(foraging_plot <- ggplot(data = foraging_combined_data,
                aes(x=Agent, y=Score)) + geom_violin(aes(fill = Agent)) + geom_point() +
    theme_minimal() +
    theme(text = element_text(size=20)) +
    scale_fill_manual(values = c("#e41a1c", "#377eb8", "#4daf4a", "#984ea3")))


## Training runs

dreamer_training <- stream_in(file("../../dreamer/logdir/foraging/foraging-train/metrics.jsonl")) %>%
  select(step, `episode/score`) %>%
  mutate(`episode/score` = str_remove_all(`episode/score`, "\\[|\\]"),
         `episode/score` = as.numeric(`episode/score`)) %>%
  drop_na() %>%
  rename(Step = step,
         Score = `episode/score`)

dreamer_training <- dreamer_training %>% mutate(Agent = rep("Dreamer-v3 (64x64 Image)"))

ppo_training <- read.csv("../data/foraging/PPORaycastForagingTraining.csv") %>%
  transmute(Agent = rep("PPO (Raycast)", nrow(.)),
            Step = Step,
            Score = Value)

foraging_combined_training <- rbind(ppo_training, dreamer_training) %>%
  mutate(Agent = as_factor(Agent),
         Step = as.numeric(Step),
         Score = as.numeric(Score))

(line_plot <- ggplot(data = foraging_combined_training,
                     aes(x=Step, y=Score, colour = Agent)) + geom_line() +
    theme_minimal() +
    theme(text = element_text(size=20)))

(smooth_plot <- ggplot(data = foraging_combined_training,
                       aes(x=Step, y=Score, colour = Agent)) + geom_smooth() +
    theme_minimal() +
    theme(text = element_text(size=20)))


################################################################################
### Button Press                                                            ####
################################################################################


dreamer <- stream_in(file("../../dreamer/logdir/button/button-eval/metrics.jsonl")) %>%
  select(`episode/score`) %>%
  mutate(`episode/score` = str_remove_all(`episode/score`, "\\[|\\]"),
         `episode/score` = as.numeric(`episode/score`)) %>%
  drop_na() %>%
  slice_sample(n = 100)

heuristic <- read.csv("../data/buttonPress/heuristicAgent100Button.csv") %>%
  select(FinalReward) %>%
  mutate(FinalReward = str_remove_all(FinalReward, "\\[|\\]"),
         FinalReward = as.numeric(FinalReward))

randomactionagent <- read.csv("../data/buttonPress/randomActionAgent100Button.csv") %>%
  select(FinalReward)

ppo <- read.csv("../data/buttonPress/PPORaycast100Button.csv") %>%
  select(FinalReward)

button_combined_data <- tibble(`Random Action Agent` = randomactionagent$FinalReward,
                        `Heuristic Agent` = heuristic$FinalReward,
                        `PPO (Raycast)` = ppo$FinalReward,
                        `Dreamer-v3 (64x64 Image)` = dreamer$`episode/score`) %>%
  pivot_longer(cols = everything(),
               names_to = "Agent",
               values_to = "Score") %>%
  mutate(Agent = as_factor(Agent))

sink("../data/buttonPress/ButtonPressTaskOmnibusTests.txt")

cat("Foraging Task: 100 runs\n")
cat("Today's Date:", Sys.Date(), "\n")
kruskal.test(Score ~ Agent, data = button_combined_data)

dunnTest(Score ~ Agent, data = button_combined_data,
         method="bonferroni")

sink()

(button_plot <- ggplot(data = button_combined_data,
                aes(x=Agent, y=Score)) + geom_violin(aes(fill = Agent)) + geom_point() +
    theme_minimal() +
    theme(text = element_text(size=20)) +
    scale_fill_manual(values = c("#e41a1c", "#377eb8", "#4daf4a", "#984ea3")))

## Training runs

dreamer_training <- stream_in(file("../../dreamer/logdir/button/button-train/metrics.jsonl")) %>%
  select(step, `episode/score`) %>%
  mutate(`episode/score` = str_remove_all(`episode/score`, "\\[|\\]"),
         `episode/score` = as.numeric(`episode/score`)) %>%
  drop_na() %>%
  rename(Step = step,
         Score = `episode/score`)

dreamer_training <- dreamer_training %>% mutate(Agent = rep("Dreamer-v3 (64x64 Image)"))

ppo_training <- read.csv("../data/buttonPress/PPORaycastButtonTraining.csv") %>%
  transmute(Agent = rep("PPO (Raycast)", nrow(.)),
            Step = Step,
            Score = Value) %>%
  filter(Step < 2000000)

button_combined_training <- rbind(ppo_training, dreamer_training) %>%
  mutate(Agent = as_factor(Agent),
         Step = as.numeric(Step),
         Score = as.numeric(Score))

(line_plot <- ggplot(data = button_combined_training,
                     aes(x=Step, y=Score, colour = Agent)) + geom_line() +
    theme_minimal() +
    theme(text = element_text(size=20)))

(smooth_plot <- ggplot(data = button_combined_training,
                       aes(x=Step, y=Score, colour = Agent)) + geom_smooth() +
    theme_minimal() +
    theme(text = element_text(size=20)) + ylim(-1.1,1))


################################################################################
### Competition                                                             ####
################################################################################


## Load all data - lots

ppo_results <- read.csv("../data/competitionAAITestbed/PPOAAITestbed2050.csv") %>%
  mutate(episode = str_remove_all(episode, ".yaml"),
         finalReward = as.numeric(finalReward),
         Agent = rep("PPO (Raycast)", nrow(.)))

ppo_training <- read.csv("../data/competitionAAITestbed/competition_raycastppo_training.csv")

heuristic_1319 <- read.csv("../data/competitionAAITestbed/Heuristic_1319.csv") %>%
  mutate(seed = rep("1319", nrow(.)))
heuristic_1357 <- read.csv("../data/competitionAAITestbed/Heuristic_1357.csv") %>%
  mutate(seed = rep("1357", nrow(.)))
heuristic_1602 <- read.csv("../data/competitionAAITestbed/Heuristic_1602.csv") %>%
  mutate(seed = rep("1602", nrow(.)))
heuristic_5281 <- read.csv("../data/competitionAAITestbed/Heuristic_5281.csv") %>%
  mutate(seed = rep("5281", nrow(.)))
heuristic_8198 <- read.csv("../data/competitionAAITestbed/Heuristic_8198.csv") %>%
  mutate(seed = rep("8198", nrow(.)))

heuristic_results <- heuristic_1319 %>%
  mutate(episode = str_remove_all(episode, ".yaml"),
         finalReward = as.numeric(str_remove_all(finalReward, "\\[|\\]")),
         Agent = rep("Heuristic Agent", nrow(.)))

all_heuristic_results <- rbind(heuristic_1319, heuristic_1357) %>%
  rbind(., heuristic_1602) %>%
  rbind(., heuristic_5281) %>%
  rbind(., heuristic_8198) %>%
  mutate(episode = str_remove_all(episode, ".yaml"),
         finalReward = as.numeric(str_remove_all(finalReward, "\\[|\\]")),
         Agent = rep("Heuristic Agent", nrow(.)))

raa_1056 <- read.csv("../data/competitionAAITestbed/RandomActionAgent_1056.csv") %>%
  mutate(seed = rep("1056", nrow(.)))
raa_1942 <- read.csv("../data/competitionAAITestbed/RandomActionAgent_1942.csv") %>%
  mutate(seed = rep("1942", nrow(.)))
raa_8812 <- read.csv("../data/competitionAAITestbed/RandomActionAgent_8812.csv") %>%
  mutate(seed = rep("8812", nrow(.)))
raa_9022 <- read.csv("../data/competitionAAITestbed/RandomActionAgent_9022.csv") %>%
  mutate(seed = rep("9022", nrow(.)))
raa_9917 <- read.csv("../data/competitionAAITestbed/RandomActionAgent_9917.csv") %>%
  mutate(seed = rep("9917", nrow(.)))

raa_results <- rbind(raa_1056, raa_1942) %>%
  rbind(., raa_8812) %>%
  rbind(., raa_9022) %>%
  rbind(., raa_9917) %>%
  mutate(episode = str_remove_all(episode, ".yaml"),
         finalReward = as.numeric(str_remove_all(finalReward, "\\[|\\]")),
         Agent = rep("Random Action Agent", nrow(.)))


pass_marks <- read.csv("../data/competitionAAITestbed/passmarks.csv") %>%
  mutate(episode = str_remove_all(episode, ".yaml"))

episode_names_1 <- pass_marks %>% filter(str_detect(episode, "[0-1][0-9]-[0-3][0-9]-01")) %>% select(episode)

episode_names_2 <- pass_marks %>% filter(str_detect(episode, "[0-1][0-9]-[0-3][0-9]-02")) %>% select(episode)

episode_names_3 <- pass_marks %>% filter(str_detect(episode, "[0-1][0-9]-[0-3][0-9]-03")) %>% select(episode)



dreamer_1_4m <- stream_in(file("../../dreamer/logdir/competition/competition-eval-ind1/metrics.jsonl")) %>%
  select(`episode/score`) %>%
  mutate(`episode/score` = str_remove_all(`episode/score`, "\\[|\\]"),
         `episode/score` = as.numeric(`episode/score`)) %>%
  drop_na() %>%
  slice_head(n = 300) %>%
  rename(finalReward = `episode/score`)

dreamer_1_4m$episode <- episode_names_1$episode

dreamer_2_4m <- stream_in(file("../../dreamer/logdir/competition/competition-eval-ind2/metrics.jsonl")) %>%
  select(`episode/score`) %>%
  mutate(`episode/score` = str_remove_all(`episode/score`, "\\[|\\]"),
         `episode/score` = as.numeric(`episode/score`)) %>%
  drop_na() %>%
  slice_head(n = 300) %>%
  rename(finalReward = `episode/score`)

dreamer_2_4m$episode <- episode_names_2$episode


dreamer_3_4m <- stream_in(file("../../dreamer/logdir/competition/competition-eval-ind3/metrics.jsonl")) %>%
  select(`episode/score`) %>%
  mutate(`episode/score` = str_remove_all(`episode/score`, "\\[|\\]"),
         `episode/score` = as.numeric(`episode/score`)) %>%
  drop_na() %>%
  slice_head(n = 300) %>%
  rename(finalReward = `episode/score`)

dreamer_3_4m$episode <- episode_names_3$episode

dreamer_4m_results <- rbind(dreamer_1_4m, dreamer_2_4m) %>%
  rbind(., dreamer_3_4m) %>%
  mutate(Agent = rep("Dreamer-v3 (64x64 Image) - 4M"))


dreamer_1_8m <- stream_in(file("../../dreamer/logdir/competition-long8m/competition-long8m-eval-ind1/metrics.jsonl")) %>%
  select(`episode/score`) %>%
  mutate(`episode/score` = str_remove_all(`episode/score`, "\\[|\\]"),
         `episode/score` = as.numeric(`episode/score`)) %>%
  drop_na() %>%
  slice_head(n = 300) %>%
  rename(finalReward = `episode/score`)

dreamer_1_8m$episode <- episode_names_1$episode

dreamer_2_8m <- stream_in(file("../../dreamer/logdir/competition-long8m/competition-long8m-eval-ind2/metrics.jsonl")) %>%
  select(`episode/score`) %>%
  mutate(`episode/score` = str_remove_all(`episode/score`, "\\[|\\]"),
         `episode/score` = as.numeric(`episode/score`)) %>%
  drop_na() %>%
  slice_head(n = 300) %>%
  rename(finalReward = `episode/score`)

dreamer_2_8m$episode <- episode_names_2$episode


dreamer_3_8m <- stream_in(file("../../dreamer/logdir/competition-long8m/competition-long8m-eval-ind3/metrics.jsonl")) %>%
  select(`episode/score`) %>%
  mutate(`episode/score` = str_remove_all(`episode/score`, "\\[|\\]"),
         `episode/score` = as.numeric(`episode/score`)) %>%
  drop_na() %>%
  slice_head(n = 300) %>%
  rename(finalReward = `episode/score`)

dreamer_3_8m$episode <- episode_names_3$episode

dreamer_8m_results <- rbind(dreamer_1_8m, dreamer_2_8m) %>%
  rbind(., dreamer_3_8m) %>%
  mutate(Agent = rep("Dreamer-v3 (64x64 Image) - 8M"))


dreamer_1_12m <- stream_in(file("../../dreamer/logdir/competition-long12m/competition-long12m-eval-ind1/metrics.jsonl")) %>%
  select(`episode/score`) %>%
  mutate(`episode/score` = str_remove_all(`episode/score`, "\\[|\\]"),
         `episode/score` = as.numeric(`episode/score`)) %>%
  drop_na() %>%
  slice_head(n = 300) %>%
  rename(finalReward = `episode/score`)

dreamer_1_12m$episode <- episode_names_1$episode

dreamer_2_12m <- stream_in(file("../../dreamer/logdir/competition-long12m/competition-long12m-eval-ind2/metrics.jsonl")) %>%
  select(`episode/score`) %>%
  mutate(`episode/score` = str_remove_all(`episode/score`, "\\[|\\]"),
         `episode/score` = as.numeric(`episode/score`)) %>%
  drop_na() %>%
  slice_head(n = 300) %>%
  rename(finalReward = `episode/score`)

dreamer_2_12m$episode <- episode_names_2$episode


dreamer_3_12m <- stream_in(file("../../dreamer/logdir/competition-long12m/competition-long12m-eval-ind3/metrics.jsonl")) %>%
  select(`episode/score`) %>%
  mutate(`episode/score` = str_remove_all(`episode/score`, "\\[|\\]"),
         `episode/score` = as.numeric(`episode/score`)) %>%
  drop_na() %>%
  slice_head(n = 300) %>%
  rename(finalReward = `episode/score`)

dreamer_3_12m$episode <- episode_names_3$episode

dreamer_12m_results <- rbind(dreamer_1_12m, dreamer_2_12m) %>%
  rbind(., dreamer_3_12m) %>%
  mutate(Agent = rep("Dreamer-v3 (64x64 Image) - 12M"))


olympics_scores <- read.csv("../data/competitionAAITestbed/olympics_scores_long.csv")


children_results <- read.csv("../data/competitionAAITestbed/children_voudouris2022_data.csv") %>%
  drop_na()  %>%
  pivot_longer(cols = !User, names_to = "episode", values_to = "finalReward") %>%
  transmute(episode = str_replace_all(episode, "\\.", "-"),
            episode = str_remove_all(episode, "P|L"),
            episode = str_replace_all(episode, "^([1-9]-)", "0\\1"),
            episode = str_replace_all(episode, "(^[0-1][0-9]-)([1-3]{0}[1-9]-)", "\\10\\2"),
            episode = str_replace_all(episode, "([1-3]$)", "0\\1"),
            Agent = rep("Human Child (Aged 6-10)"),
            finalReward = finalReward) 

children_results_all <- read.csv("../data/competitionAAITestbed/children_voudouris2022_data.csv") %>%
  drop_na()  %>%
  pivot_longer(cols = !User, names_to = "episode", values_to = "finalReward") %>%
  transmute(episode = str_replace_all(episode, "\\.", "-"),
            episode = str_remove_all(episode, "P|L"),
            episode = str_replace_all(episode, "^([1-9]-)", "0\\1"),
            episode = str_replace_all(episode, "(^[0-1][0-9]-)([1-3]{0}[1-9]-)", "\\10\\2"),
            episode = str_replace_all(episode, "([1-3]$)", "0\\1"),
            Agent = rep("Human Child (Aged 6-10)"),
            finalReward = finalReward,
            id = User) 



## Combine all data


all_results <- bind_rows(ppo_results, heuristic_results) %>%
  bind_rows(., raa_results) %>%
  bind_rows(., dreamer_4m_results) %>%
  bind_rows(., dreamer_8m_results) %>%
  bind_rows(., dreamer_12m_results) %>%
  bind_rows(., olympics_scores) %>%
  bind_rows(., children_results) %>%
  left_join(., pass_marks, by = c("episode" = "episode")) %>%
  separate(episode, c('Level', 'Task', 'Variant')) %>%
  mutate(Level = ifelse(Level == '01'|Level == '1', 'Food Retrieval',
                        ifelse(Level == '02'|Level == '2', 'Preferences',
                               ifelse(Level == '03'|Level == '3', 'Static Obstacles',
                                      ifelse(Level == '04'|Level == '4', 'Avoidance',
                                             ifelse(Level == '05'|Level == '5', 'Spatial Reasoning\nand Support',
                                                    ifelse(Level == '06'|Level == '6', 'Generalisation',
                                                           ifelse(Level == '07'|Level == '7', 'Internal Modelling',
                                                                  ifelse(Level == '08'|Level == '8', 'Object Permanence\nand Working Memory',
                                                                         ifelse(Level == '09'|Level == '9', 'Numerosity and\nAdvanced Preferences',
                                                                                ifelse(Level == '10', 'Causal Reasoning', NA)))))))))),
         Pass = ifelse(finalReward >= passMark, 1, 0))

## Plot data

bar_plot_results <- all_results %>% group_by(Level, Agent) %>%
  #filter(Agent == 'ironbar' | Agent == 'Trrrrr' | Agent == 'sirius' | Agent == 'Melflo' | Agent == 'sungbinchoi' | Agent == 'BronzeBlood' | Agent == 'ppo' | Agent == 'randomactionagent' | Agent == 'braitenberg' | Agent == 'Dreamer-v3 (64x64 Image)') %>%
  filter(Agent == 'ironbar' | Agent == 'Trrrrr' | Agent == 'PPO (Raycast)' | Agent == 'Random Action Agent' | Agent == 'Heuristic Agent' | Agent == 'Dreamer-v3 (64x64 Image) - 4M' | Agent == 'Dreamer-v3 (64x64 Image) - 8M' | Agent == 'Dreamer-v3 (64x64 Image) - 12M' | Agent == "Human Child (Aged 6-10)") %>%
  summarise(`Proportion Passed` = sum(Pass)/n())

bar_plot_results$Level <- factor(bar_plot_results$Level,
                                 levels = c('Food Retrieval',
                                            'Preferences',
                                            'Static Obstacles',
                                            'Avoidance',
                                            'Spatial Reasoning\nand Support',
                                            'Generalisation',
                                            'Internal Modelling',
                                            'Object Permanence\nand Working Memory',
                                            'Numerosity and\nAdvanced Preferences',
                                            'Causal Reasoning'))
bar_plot_agents <- bar_plot_results %>%
  filter(Agent == 'Heuristic Agent' | Agent == 'ironbar' | Agent == 'Trrrrr' | Agent == 'PPO (Raycast)' | Agent == 'Dreamer-v3 (64x64 Image) - 4M' | Agent == 'Dreamer-v3 (64x64 Image) - 8M' | Agent == 'Dreamer-v3 (64x64 Image) - 12M')

bar_plot_agents$Agent <- factor(bar_plot_agents$Agent,
                                levels = c(#'Random Action Agent',
                                  'Heuristic Agent',
                                  #'sungbinchoi',
                                  #'Melflo',
                                  #'BronzeBlood',
                                  #'sirius',
                                  'ironbar',
                                  'Trrrrr',
                                  'PPO (Raycast)',
                                  'Dreamer-v3 (64x64 Image) - 4M',
                                  'Dreamer-v3 (64x64 Image) - 8M',
                                  'Dreamer-v3 (64x64 Image) - 12M'
                                  #"Human Child (Aged 6-10)"
                                ))

bar_plot_references <- bar_plot_results %>%
  filter(Agent == 'Random Action Agent' | Agent == 'Human Child (Aged 6-10)') %>%
  mutate(Reference = Agent)

bar_plot_references$Reference <- factor(bar_plot_references$Reference,
                                        levels = c('Random Action Agent',
                                                   #'Heuristic Agent',
                                                   #'sungbinchoi',
                                                   #'Melflo',
                                                   #'BronzeBlood',
                                                   #'sirius',
                                                   #'ironbar',
                                                   #'Trrrrr',
                                                   #'PPO (Raycast)',
                                                   #'Dreamer-v3 (64x64 Image)',
                                                   "Human Child (Aged 6-10)"
                                        ))

(plot <- ggplot() + 
    geom_bar(data = bar_plot_agents,
             aes(x=Level, y=`Proportion Passed`, fill = Agent),
             position='dodge', stat='identity') +
    theme_minimal() +
    theme(text = element_text(size=20), axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),panel.grid = element_blank()) + ylim(0,1) +
    geom_vline(xintercept = seq(0.5, length(bar_plot_agents$Level), by = 1), color="gray", size=.5, alpha=1) +
    geom_segment(data = bar_plot_references,
                 aes(
                   x = as.numeric(Level) - .45,
                   xend = as.numeric(Level) + .45,
                   y = `Proportion Passed`,
                   yend = `Proportion Passed`,
                   colour = Reference,
                   linetype = Reference
                 ), linewidth = 1) +
    #+
    scale_fill_manual(values = c("#377eb8","#ff7f00", "#ffff33", "#4daf4a", "#756bb1",  "#984ea3", "#7a0177"))
  #labs(caption = "Agents played all 90 tasks in each level. Children (n=59) played a randomly selected subset of 4 tasks in each level.") +  
  # scale_fill_manual(values=c("#0073e6",
  #                            "#e6308a",
  #                            "#b51963",
  #                            "#9efd38",
  #                            "#006600",
  #                            "#138808",
  #                             "#568203")) + 
  +  scale_color_manual(values = c(
    "#1a237e",
    "#000000"
  ))
)

## Training

dreamer_training_comp_4m <- stream_in(file("../../dreamer/logdir/competition/competition-train/metrics.jsonl")) %>%
  select(step, `episode/score`) %>%
  mutate(`episode/score` = str_remove_all(`episode/score`, "\\[|\\]"),
         `episode/score` = as.numeric(`episode/score`)) %>%
  drop_na() %>%
  rename(Step = step,
         Score = `episode/score`)

dreamer_training_comp_8m <- stream_in(file("../../dreamer/logdir/competition-long8m/competition-train-long8m/metrics.jsonl")) %>%
  select(step, `episode/score`) %>%
  mutate(`episode/score` = str_remove_all(`episode/score`, "\\[|\\]"),
         `episode/score` = as.numeric(`episode/score`)) %>%
  drop_na() %>%
  rename(Step = step,
         Score = `episode/score`)

dreamer_training_comp_8m <- dreamer_training_comp_8m[-1,] #remove the first row which is the first step

dreamer_training_comp_12m <- stream_in(file("../../dreamer/logdir/competition-long12m/competition-train-long12m/metrics.jsonl")) %>%
  select(step, `episode/score`) %>%
  mutate(`episode/score` = str_remove_all(`episode/score`, "\\[|\\]"),
         `episode/score` = as.numeric(`episode/score`)) %>%
  drop_na() %>%
  rename(Step = step,
         Score = `episode/score`)

dreamer_training_comp_12m <- dreamer_training_comp_12m[-1,] #remove the first row which is the first step

dreamer_training_comp <- bind_rows(dreamer_training_comp_4m, dreamer_training_comp_8m) %>% 
  bind_rows(., dreamer_training_comp_12m) %>%
  mutate(Agent = rep("Dreamer-v3 (64x64 Image)"))

ppo_training_comp <- read.csv("../data/competitionAAITestbed/competition_raycastppo_training.csv") %>%
  transmute(Agent = rep("PPO (Raycast)", nrow(.)),
            Step = Step,
            Score = Value)


combined <- rbind(dreamer_training_comp, ppo_training_comp) %>%
  mutate(Agent = as_factor(Agent),
         Step = as.numeric(Step),
         Score = as.numeric(Score))

(line_plot <- ggplot(data = combined,
                     aes(x=Step, y=Score, colour = Agent)) + geom_line() +
    theme_minimal() +
    theme(text = element_text(size=20)))

(smooth_plot <- ggplot(data = combined,
                       aes(x=Step, y=Score, colour = Agent)) + geom_smooth() +
    theme_minimal() +
    theme(text = element_text(size=20)) + ylim(-1.1,1))


### Differences

raa_results_all <- raa_results %>% rename(id = seed)
dreamer_4m_results_all <- dreamer_4m_results %>% mutate(id = "Dreamer-v3 (64x64 Image) - 4M")
dreamer_8m_results_all <- dreamer_8m_results %>% mutate(id = "Dreamer-v3 (64x64 Image) - 8M")
dreamer_12m_results_all <- dreamer_12m_results %>% mutate(id = "Dreamer-v3 (64x64 Image) - 12M")
ppo_results_all <- ppo_results %>% mutate(id = "PPO (Raycast)")
olympics_scores_all <- olympics_scores %>%
  filter(Agent == "Trrrrr" | Agent == "ironbar") %>%
  mutate(id = Agent)


all_results_lmer <- bind_rows(raa_results_all, all_heuristic_results) %>%
  bind_rows(., dreamer_4m_results_all) %>%
  bind_rows(., dreamer_8m_results_all) %>%
  bind_rows(., dreamer_12m_results_all) %>%
  bind_rows(., ppo_results_all) %>%
  bind_rows(., olympics_scores_all) %>%
  bind_rows(., children_results_all) %>%
  left_join(., pass_marks, by = c("episode" = "episode")) %>%
  separate(episode, c('Level', 'Task', 'Variant')) %>%
  mutate(Level = ifelse(Level == '01'|Level == '1', 'Food Retrieval',
                        ifelse(Level == '02'|Level == '2', 'Preferences',
                               ifelse(Level == '03'|Level == '3', 'Static Obstacles',
                                      ifelse(Level == '04'|Level == '4', 'Avoidance',
                                             ifelse(Level == '05'|Level == '5', 'Spatial Reasoning\nand Support',
                                                    ifelse(Level == '06'|Level == '6', 'Generalisation',
                                                           ifelse(Level == '07'|Level == '7', 'Internal Modelling',
                                                                  ifelse(Level == '08'|Level == '8', 'Object Permanence\nand Working Memory',
                                                                         ifelse(Level == '09'|Level == '9', 'Numerosity and\nAdvanced Preferences',
                                                                                ifelse(Level == '10', 'Causal Reasoning', NA)))))))))),
         Pass = ifelse(finalReward >= passMark, 1, 0),
         Instance = paste0(Level, "-", Task, "-", Variant),
         Agent = haven::as_factor(Agent),
         Level = haven::as_factor(Level))



model <- glmer(Pass ~ Agent + Level + (1|id), data = all_results_lmer, family = binomial(link = "logit"),
               control = glmerControl(optimizer = "bobyqa"),
               nAGQ = 10)

sink("../data/competitionAAITestbed/competitionGLMMTest.txt")

print(model, corr=FALSE)

summary(model)

exp(fixef(model))

se <- sqrt(diag(vcov(model)))

(tab <- cbind(Est = fixef(model), LL = fixef(model) - 1.96 * se, UL = fixef(model) + 1.96 *
                se))

odds_ratios_CIs <- exp(tab)

sink()

odds_ratios_CIs_tib <- tibble("Factor" = row.names(odds_ratios_CIs),
                              "Mean Estimate" = odds_ratios_CIs[,1],
                              "Lower Bound" = odds_ratios_CIs[,2],
                              "Upper Bound" = odds_ratios_CIs[,3]) %>%
  mutate(Type = ifelse(str_detect(Factor, "Level"), "Level", 
                       ifelse(str_detect(Factor, "Agent"), "Agent", "Intercept")),
         Factor = str_remove_all(Factor, "Level|Agent"))


(gg0 <- (ggplot(odds_ratios_CIs_tib,
                aes(x = `Mean Estimate`, y = `Factor`))
         + geom_pointrange(aes(xmin = `Lower Bound`, xmax = `Upper Bound`))
         + geom_vline(xintercept = 0, lty = 2)
))

## all results averages

proportions_passed_overall <- all_results %>%
  mutate(Agent = haven::as_factor(Agent)) %>%
  group_by(Agent) %>%
  summarise(`Proportion Passed` = sum(Pass)/n()) %>%
  arrange(desc(`Proportion Passed`))

