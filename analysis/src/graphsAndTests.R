### Graph Plotting and Analysis for Animal-AI 3 Paper
### Author: K. Voudouris, 2024 (C)
### Version: 4.4.1 (Race for Your Life)

### Packages and Preamble

library(FSA)
library(jsonlite)
library(lme4)
library(svglite)
library(tidyverse)

options(scipen = 999)
set.seed(2023)

# Set working directory to path to this file

################################################################################
### Foraging Tree                                                           ####
################################################################################

dreamer <- read.csv("../../dreamer/results/foragingTask-eval.csv")

heuristic <- read.csv("../../random_heuristic_agents/results/foragingTask/heuristic.csv") %>%
  select(FinalReward) %>%
  transmute(finalReward = str_remove_all(FinalReward, "\\[|\\]"),
            finalReward = as.numeric(finalReward))

random <- read.csv("../../random_heuristic_agents/results/foragingTask/random.csv") %>%
  select(FinalReward) %>%
  transmute(finalReward = str_remove_all(FinalReward, "\\[|\\]"),
            finalReward = as.numeric(finalReward))

ppo <- read.csv("../../ppo/results/foragingTask/ppo/all.csv") %>%
  select(episode_reward) %>%
  rename(finalReward = episode_reward)

foraging_combined_data <- tibble(`Random Action Agent` = random$finalReward,
                                 `Heuristic Agent` = heuristic$finalReward,
                                 `PPO` = ppo$finalReward,
                                 `Dreamer-v3` = dreamer$finalReward) %>%
  pivot_longer(cols = everything(),
               names_to = "Agent",
               values_to = "Score") %>%
  mutate(Agent = as_factor(Agent))

sink("../outputs/foragingTaskOmnibusTests.txt")

cat("Foraging Task: 100 runs\n")
cat("Today's Date:", Sys.Date(), "\n")
kruskal.test(Score ~ Agent, data = foraging_combined_data)

dunnTest(Score ~ Agent, data = foraging_combined_data,
         method="bonferroni")

sink()

(foraging_plot <- ggplot(data = foraging_combined_data,
                         aes(x=Agent, y=Score)) + geom_violin(aes(fill = Agent), scale = "width") + geom_point() +
    theme_minimal() +
    theme(text = element_text(size=20),
          axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
    scale_color_manual(values = c("#e41a1c", "#377eb8", "#984ea3", "#4daf4a")) +
    scale_fill_manual(values = c("#e41a1c", "#377eb8", "#984ea3", "#4daf4a")))

ggsave("foragingTaskViolinPlot.svg", plot = foraging_plot, device = "svg", path = "../figures/", width = 5000, height = 2800, units = "px")

## Training runs

dreamer_training <- read.csv("../../dreamer/results/foragingTask-training.csv") %>% 
  transmute(Agent = rep("Dreamer-v3"),
            Step = Step,
            Score = Score_movingavg)

ppo_training <- read.csv("../../ppo/results/foragingTask/ppo/training.csv") %>%
  transmute(Agent = rep("PPO", nrow(.)),
            Step = (Step+1)*(1000000/(max(.$Step)+1)),
            Score = PPO..foraging.training..1M.steps..aai_timescale.1...rollout.ep_rew_mean)

foraging_combined_training <- rbind(ppo_training, dreamer_training) %>%
  mutate(Agent = factor(Agent, levels = c("Dreamer-v3", "PPO")),
         Step = as.numeric(Step),
         Score = as.numeric(Score))

(foraging_line_plot <- ggplot(data = foraging_combined_training,
                              aes(x=Step, y=Score, colour = Agent)) + geom_line(size = 1.5) +
    scale_color_manual(values = c("#984ea3", "#4daf4a")) +
    theme_minimal() +
    theme(text = element_text(size=20)))

ggsave("foragingTaskTrainingPlot.svg", plot = foraging_line_plot, device = "svg", path = "../figures/", width = 5000, height = 2800, units = "px")

################################################################################
### Operant Chamber Task                                                    ####
################################################################################


dreamer_basic <- read.csv("../../dreamer/results/operantChamber-eval.csv")

dreamer_curriculum <- read.csv("../../dreamer/results/operantChamberCurriculum-eval.csv")

heuristic <- read.csv("../../random_heuristic_agents/results/operantChamberTask/heuristic.csv") %>%
  select(FinalReward) %>%
  transmute(finalReward = str_remove_all(FinalReward, "\\[|\\]"),
            finalReward = as.numeric(finalReward))

random <- read.csv("../../random_heuristic_agents/results/operantChamberTask/random.csv") %>%
  select(FinalReward) %>%
  mutate(finalReward = str_remove_all(FinalReward, "\\[|\\]"),
         finalReward = as.numeric(finalReward))

ppo_basic <- read.csv("../../ppo/results/operantChamber/ppo/all.csv") %>%
  transmute(finalReward = as.numeric(episode_reward))

ppo_curriculum <- read.csv("../../ppo/results/operantChamber/ppo/all-curriculum.csv") %>%
  transmute(finalReward = as.numeric(episode_reward))

operantchamber_combined_data <- tibble(`Random Action Agent` = random$finalReward,
                                       `Heuristic Agent` = heuristic$finalReward,
                                       `PPO` = ppo_basic$finalReward,
                                       `PPO Curriculum` = ppo_curriculum$finalReward,
                                       `Dreamer-v3` = dreamer_basic$finalReward,
                                       `Dreamer-v3 Curriculum` = dreamer_curriculum$finalReward) %>%
  pivot_longer(cols = everything(),
               names_to = "Agent",
               values_to = "Score") %>%
  mutate(Agent = as_factor(Agent))

sink("../outputs/operantChamberTaskOmnibusTests.txt")

cat("Operant Chamber Task: 100 runs\n")
cat("Today's Date:", Sys.Date(), "\n")
kruskal.test(Score ~ Agent, data = operantchamber_combined_data)

dunnTest(Score ~ Agent, data = operantchamber_combined_data,
         method="bonferroni")

sink()

(operantChamber_plot <- ggplot(data = operantchamber_combined_data,
                               aes(x=Agent, y=Score)) + geom_violin(aes(fill = Agent), scale = "width") + geom_point() +
    theme_minimal() +
    theme(text = element_text(size=20),
          axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
    scale_fill_manual(values = c("#e41a1c", "#377eb8", "#984ea3", "#984ef7", "#4daf4a", "#4dee4a")))

ggsave("operantChamberViolinPlot.svg", plot = operantChamber_plot, device = "svg", path = "../figures/", width = 5000, height = 2800, units = "px")

## Training runs

dreamer_basic_training <- read.csv("../../dreamer/results/operantChamber-training.csv") %>%
  transmute(Agent = rep("Dreamer-v3"),
            Step = Step,
            Score = Score_movingavg)

dreamer_curriculum_training <- read.csv("../../dreamer/results/operantChamberCurriculum-training.csv") %>%
  transmute(Agent = rep("Dreamer-v3"),
            Step = Step,
            Score = Score_movingavg)

ppo_basic_training <- read.csv("../../ppo/results/operantChamber/ppo/training.csv") %>%
  transmute(Agent = rep("PPO", nrow(.)),
            Step = (Step+1)*(2000000/(max(.$Step)+1)),
            Score = PPO..default...operant.training..aai_timescale.1..2M.steps...rollout.ep_rew_mean)

ppo_curriculum_training_data <- read.csv("../../ppo/results/operantChamber/ppo/training-curriculum.csv") %>%
  select(!(ends_with("_MIN") | ends_with("_MAX")))

rewards <- c(ppo_curriculum_training_data$PPO..default...operantCurriculum.A..timescale.1..400K.timesteps...rollout.ep_rew_mean,
             ppo_curriculum_training_data$PPO..default...operantCurriculum.B..timescale.1..400K.timesteps...rollout.ep_rew_mean,
             ppo_curriculum_training_data$PPO..default...operantCurriculum.C..timescale.1..400K.timesteps...rollout.ep_rew_mean,
             ppo_curriculum_training_data$PPO..default...operantCurriculum.D..timescale.1..400K.timesteps...rollout.ep_rew_mean,
             ppo_curriculum_training_data$PPO..default...operantCurriculum.E..timescale.1..400K.timesteps...rollout.ep_rew_mean
)

ppo_curriculum_training <- tibble(Agent = rep("PPO", 1000),
                                  Step = seq(2000, 2000000, 2000),
                                  Score = rewards)

operantchamber_combined_training_basic <- bind_rows(ppo_basic_training, 
                                                    dreamer_basic_training) %>%
  mutate(Agent = factor(Agent, levels=c("Dreamer-v3", "PPO")),
         Step = as.numeric(Step),
         Score = as.numeric(Score))

operantchamber_combined_training_curriculum <- bind_rows(ppo_curriculum_training, 
                                                         dreamer_curriculum_training) %>%
  mutate(Agent = factor(Agent, levels=c("Dreamer-v3", "PPO")),
         Step = as.numeric(Step),
         Score = as.numeric(Score))

background_colour_rects <- data.frame(start = seq(0, 1600000, 400000), 
                                      end = seq(400000, 2000000, 400000), 
                                      level = c ("A", "B", "C", "D", "E"))

colours <- c("grey","white","grey","white","grey")
names(colours) <- background_colour_rects$level

(operantChamber_training_plot_basic <- ggplot()  + 
    scale_fill_manual(values=colours) + geom_line(data = operantchamber_combined_training_basic,
                                                  aes(x=Step, y=Score, colour = Agent), size = 1.5)  +
    scale_color_manual(values = c("#984ea3", "#4daf4a")) +
    # annotate("segment", x=0, xend=2000000, y=mean(random$finalReward), yend=mean(random$finalReward), color = "black", linetype="dashed") +
    # annotate("segment", x=0, xend=2000000, y=mean(heuristic$finalReward), yend=mean(heuristic$finalReward), color = "black", linetype="dashed") +
    # #annotate("label", x = 2050000, y = tail(ppo_basic_training$Score, n=1), label = "PPO") +
    # #annotate("label", x = 2050000, y = tail(dreamer_basic_training$Score, n=1), label = "Dreamer") +
    # annotate("label", x = 250000, y = mean(random$finalReward)+0.05, label = "Random") +
    # annotate("label", x = 250000, y = mean(heuristic$finalReward)+0.05, label = "Heuristic") +
    theme_minimal() +
    theme(text = element_text(size=20)))

ggsave("operantChamberBasicTrainingPlot.svg", plot = operantChamber_training_plot_basic, device = "svg", path = "../figures/", width = 5000, height = 2800, units = "px")

(operantChamber_training_plot_curriculum <- ggplot()  + 
    geom_rect(data = background_colour_rects, aes(xmin = start, xmax = end, ymin = -1, ymax = 1, fill = level), alpha = 0.5, inherit.aes = FALSE) +
    scale_color_manual(values = c("#984ef7", "#4dee4a")) +
    scale_fill_manual(values=colours, guide="none") + geom_line(data = operantchamber_combined_training_curriculum,
                                                                aes(x=Step, y=Score, colour = Agent), size = 1.5) +
    # annotate("segment", x=0, xend=2000000, y=mean(random$finalReward), yend=mean(random$finalReward), color = "black", linetype="dashed") +
    # annotate("segment", x=0, xend=2000000, y=mean(heuristic$finalReward), yend=mean(heuristic$finalReward), color = "black", linetype="dashed") +
    # #annotate("label", x = 2050000, y = tail(ppo_basic_training$Score, n=1), label = "PPO") +
    # #annotate("label", x = 2050000, y = tail(dreamer_basic_training$Score, n=1), label = "Dreamer") +
    # annotate("label", x = 1800000, y = mean(random$finalReward)+0.05, label = "Random") +
    # annotate("label", x = 1800000, y = mean(heuristic$finalReward)+0.05, label = "Heuristic") +
    theme_minimal() +
    theme(text = element_text(size=20)))

ggsave("operantChamberCurriculumTrainingPlot.svg", plot = operantChamber_training_plot_curriculum, device = "svg", path = "../figures/", width = 5000, height = 2800, units = "px")


################################################################################
### Competition                                                             ####
################################################################################

## Load all data

pass_marks <- read.csv("../data/competitionAAITestbed/passmarks.csv") %>%
  mutate(episode = str_remove_all(episode, ".yaml"))

olympics_scores <- read.csv("../data/competitionAAITestbed/olympics_scores_long.csv")

episode_names_1 <- pass_marks %>% filter(str_detect(episode, "[0-1][0-9]-[0-3][0-9]-01")) %>% select(episode)

episode_names_2 <- pass_marks %>% filter(str_detect(episode, "[0-1][0-9]-[0-3][0-9]-02")) %>% select(episode)

episode_names_3 <- pass_marks %>% filter(str_detect(episode, "[0-1][0-9]-[0-3][0-9]-03")) %>% select(episode)

dreamer_results <- read.csv("../../dreamer/results/competition-eval.csv") %>%
  mutate(Agent = rep("Dreamer-v3", nrow(.)))

ppo_results <- read.csv("../../ppo/results/competition/ppo/all.csv") %>%
  transmute(episode = str_remove_all(arena_name, ".yaml"),
            finalReward = as.numeric(episode_reward),
            Agent = rep("PPO", nrow(.)))

heuristic_results <- read.csv("../../random_heuristic_agents/results/competition/heuristicAgent.csv") %>%
  transmute(episode = str_remove_all(episode, ".yaml"),
            finalReward = str_remove_all(finalReward, "\\[|\\]"),
            finalReward = as.numeric(finalReward),
            Agent = rep("Heuristic Agent", nrow(.)))

random_results <- read.csv("../../random_heuristic_agents/results/competition/randomAgent.csv") %>%
  transmute(episode = str_remove_all(episode, ".yaml"),
            finalReward = str_remove_all(finalReward, "\\[|\\]"),
            finalReward = as.numeric(finalReward),
            Agent = rep("Random Action Agent", nrow(.)))

children_results <- read.csv("../data/competitionAAITestbed/children_voudouris2022_data.csv") %>%
  drop_na()  %>%
  pivot_longer(cols = !User, names_to = "episode", values_to = "finalReward") %>%
  transmute(episode = str_replace_all(episode, "\\.", "-"),
            episode = str_remove_all(episode, "P|L"),
            episode = str_replace_all(episode, "^([1-9]-)", "0\\1"),
            episode = str_replace_all(episode, "(^[0-1][0-9]-)([1-3]{0}[1-9]-)", "\\10\\2"),
            episode = str_replace_all(episode, "([1-3]$)", "0\\1"),
            Agent = rep("Children (Aged 6-10)"),
            finalReward = finalReward) 

children_results_all <- read.csv("../data/competitionAAITestbed/children_voudouris2022_data.csv") %>%
  drop_na()  %>%
  pivot_longer(cols = !User, names_to = "episode", values_to = "finalReward") %>%
  transmute(episode = str_replace_all(episode, "\\.", "-"),
            episode = str_remove_all(episode, "P|L"),
            episode = str_replace_all(episode, "^([1-9]-)", "0\\1"),
            episode = str_replace_all(episode, "(^[0-1][0-9]-)([1-3]{0}[1-9]-)", "\\10\\2"),
            episode = str_replace_all(episode, "([1-3]$)", "0\\1"),
            Agent = rep("Children (Aged 6-10)"),
            finalReward = finalReward,
            id = User) 

## Combine all data

all_results <- bind_rows(ppo_results, heuristic_results) %>%
  bind_rows(., random_results) %>%
  bind_rows(., dreamer_results) %>%
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
  filter(Agent == 'ironbar' | Agent == 'Trrrrr' | Agent == 'PPO' | Agent == 'Random Action Agent' | Agent == 'Heuristic Agent' | Agent == 'Dreamer-v3' | Agent == "Children (Aged 6-10)") %>%
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
  filter(Agent == 'Heuristic Agent' | Agent == 'ironbar' | Agent == 'Trrrrr' | Agent == 'PPO' | Agent == 'Dreamer-v3')

bar_plot_agents$Agent <- factor(bar_plot_agents$Agent,
                                levels = c('Heuristic Agent',
                                           #'sungbinchoi',
                                           #'Melflo',
                                           #'BronzeBlood',
                                           #'sirius',
                                           'ironbar',
                                           'Trrrrr',
                                           'PPO',
                                           'Dreamer-v3' #,
                                ))

bar_plot_references <- bar_plot_results %>%
  filter(Agent == 'Random Action Agent' | Agent == 'Children (Aged 6-10)') %>%
  mutate(Reference = Agent)

bar_plot_references$Reference <- factor(bar_plot_references$Reference,
                                        levels = c('Random Action Agent',
                                                   "Children (Aged 6-10)"
                                        ))

(competition_bar_plot <- ggplot() + 
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
    scale_fill_manual(values = c("#377eb8","#ff7f00", "#ffff33", "#984ea3", "#4daf4a")) +  scale_color_manual(values = c(
      "#1a237e",
      "#000000"
    ))
)

ggsave("competitionBarPlot.svg", plot = competition_bar_plot, device = "svg", path = "../figures/", width = 5000, height = 2800, units = "px")

## Training

dreamer_competition_training <- read.csv("../../dreamer/results/competition-training.csv") %>%
  transmute(Agent = rep("Dreamer-v3"),
            Step = Step,
            Score = Score_movingavg)

ppo_competition_training_scores <- read.csv("../../ppo/results/competition/ppo/training.csv") %>%
  select(!(ends_with("_MIN") | ends_with("_MAX")))

rewards <- c(ppo_competition_training_scores$PPO..default...level.1..aai_timescale.300..2M.time.steps...rollout.ep_rew_mean,
             ppo_competition_training_scores$PPO..default...levels.1.2..aai_timescale.300..2M.time.steps...rollout.ep_rew_mean,
             ppo_competition_training_scores$PPO..default...levels.1.3..aai_timescale.300..2M.time.steps...rollout.ep_rew_mean,
             ppo_competition_training_scores$PPO..default...levels.1.4..aai_timescale.300..2M.time.steps...rollout.ep_rew_mean,
             ppo_competition_training_scores$PPO..default...levels.1.5..aai_timescale.300..2M.time.steps...rollout.ep_rew_mean,
             ppo_competition_training_scores$PPO..default...levels.1.6..aai_timescale.300..2M.time.steps...rollout.ep_rew_mean,
             ppo_competition_training_scores$PPO..default...levels.1.7..aai_timescale.300..2M.time.steps...rollout.ep_rew_mean,
             ppo_competition_training_scores$PPO..default...levels.1.8..aai_timescale.300..2M.time.steps...rollout.ep_rew_mean,
             ppo_competition_training_scores$PPO..default...levels.1.9..aai_timescale.300..2M.time.steps...rollout.ep_rew_mean,
             ppo_competition_training_scores$PPO..default...levels.1.10..aai_timescale.300..2M.time.steps...rollout.ep_rew_mean,
             ppo_competition_training_scores$PPO..default...levels.1.10..aai_timescale.300..5M.time.steps...rollout.ep_rew_mean
)

rewards <- rewards[!is.na(rewards)]

ppo_competition_training <- tibble(Agent = rep("PPO", length(rewards)),
                                   Step = seq(0, 25000000, 25000000/(length(rewards)-1)),
                                   Score = rewards)

competition_combined_training <- bind_rows(ppo_competition_training, 
                                           dreamer_competition_training) %>%
  mutate(Agent = factor(Agent, levels=c("Dreamer-v3", "PPO")),
         Step = as.numeric(Step),
         Score = as.numeric(Score))


background_colour_rects <- data.frame(start = seq(0, 18000000, 2000000), 
                                      end = c(seq(2000000, 18000000, 2000000), 25000000), 
                                      level = c ("L1", "L1-2", "L1-3", "L1-4", "L1-5", "L1-6", "L1-7", "L1-8", "L1-9", "L1-10"))

colours <- c("white", "grey","white","grey","white","grey","white","grey","white","grey")
names(colours) <- background_colour_rects$level

(line_plot_competition <- ggplot()  + 
    geom_rect(data = background_colour_rects, aes(xmin = start, xmax = end, ymin = -1, ymax = 5, fill = level), alpha = 0.5, inherit.aes = FALSE) +
    scale_fill_manual(values=colours, guide="none") + geom_line(data = competition_combined_training,
                                                                aes(x=Step, y=Score, colour = Agent), size = 1.5)  +
    scale_color_manual(values = c("#984ea3", "#4daf4a")) +
    theme_minimal() +
    theme(text = element_text(size=20)))

ggsave("competitionTrainingPlot.svg", plot = line_plot_competition, device = "svg", path = "../figures/", width = 5000, height = 2800, units = "px")

### Differences

random_results_id <- random_results %>% mutate(id = "Random Action Agent")

heuristic_results_id <- heuristic_results %>% mutate(id = "Heuristic Agent")

dreamer_results_id <- dreamer_results %>% mutate(id = "Dreamer-v3")

ppo_results_id <- ppo_results %>% mutate(id = "PPO")

olympics_scores_all <- olympics_scores %>%
  filter(Agent == "Trrrrr" | Agent == "ironbar") %>%
  mutate(id = Agent)

all_results_lmer <- bind_rows(random_results_id, heuristic_results_id) %>%
  bind_rows(., dreamer_results_id) %>%
  bind_rows(., ppo_results_id) %>%
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

sink("../outputs/competitionGLMMTest.txt")

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


(competition_forest_plot <- (ggplot(odds_ratios_CIs_tib,
                                    aes(x = `Mean Estimate`, y = `Factor`))
                             + geom_pointrange(aes(xmin = `Lower Bound`, xmax = `Upper Bound`))
                             + geom_vline(xintercept = 0, lty = 2)
))

ggsave("competitionForestPlot.svg", plot = competition_forest_plot, device = "svg", path = "../figures/", width = 5000, height = 2800, units = "px")

## all results averages

proportions_passed_overall <- all_results %>%
  mutate(Agent = haven::as_factor(Agent)) %>%
  group_by(Agent) %>%
  summarise(`Proportion Passed` = sum(Pass)/n()) %>%
  arrange(desc(`Proportion Passed`))

write.csv(proportions_passed_overall, "../outputs/overallCompetitionLeaderboard.csv", row.names = FALSE)

cat(".\n.\n.\nFinished!")
