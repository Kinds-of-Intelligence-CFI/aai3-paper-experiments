### Dreamer Data Cleaning for Animal-AI Paper
### Author: K. Voudouris, 2024 (C)
### Version: 4.4.1 (Race for Your Life)

### Packages and Preamble

library(jsonlite)
library(tidyverse)
library(zoo)

options(scipen = 999)
set.seed(2023)

movingAverage <- function(path, width = 500){
  data <- stream_in(file(path)) %>%
    select(step, `episode/score`) %>%
    mutate(Lag0 = as.numeric(str_remove_all(`episode/score`, "\\[|\\]"))) %>%
    transmute(Step = step,
              Score_raw = Lag0,
              Score_movingavg = zoo::rollapply(Lag0, width=width, FUN=function(x) mean(x, na.rm=TRUE), fill=NA, partial=TRUE))
  
}

# Set working directory to path to this file

# Foraging Task

foraging_task <- stream_in(file("../logdir/foraging/foraging-eval/metrics.jsonl")) %>%
  select(`episode/score`) %>%
  transmute(finalReward = str_remove_all(`episode/score`, "\\[|\\]"),
         finalReward = as.numeric(finalReward)) %>%
  drop_na() %>%
  slice_sample(n = 100)

write.csv(foraging_task, "foragingTask-eval.csv", row.names = FALSE)

foraging_task_training <- stream_in(file("..//logdir/foraging/foraging-train/metrics.jsonl")) %>%
  select(step, `episode/score`) %>%
  mutate(`episode/score` = str_remove_all(`episode/score`, "\\[|\\]"),
         `episode/score` = as.numeric(`episode/score`)) %>%
  drop_na() %>%
  rename(Step = step,
         Score = `episode/score`)

write.csv(foraging_task_training, "foragingTask-training.csv", row.names = FALSE)

# Operant Chamber Task

operantChamber_basic <- stream_in(file("..//logdir/operantChamber/operantChamber-eval/metrics.jsonl")) %>%
  select(`episode/score`) %>%
  transmute(finalReward = str_remove_all(`episode/score`, "\\[|\\]"),
            finalReward = as.numeric(finalReward)) %>%
  drop_na() %>%
  slice_sample(n = 100)

write.csv(operantChamber_basic, "operantChamber-eval.csv", row.names = FALSE)

operantChamber_curriculum <- stream_in(file("../logdir/operantChamber/operantChamberCurriculum-eval/metrics.jsonl")) %>%
  select(`episode/score`) %>%
  mutate(finalReward = str_remove_all(`episode/score`, "\\[|\\]"),
         finalReward = as.numeric(finalReward)) %>%
  drop_na() %>%
  slice_sample(n = 100)

write.csv(operantChamber_curriculum, "operantChamberCurriculum-eval.csv", row.names = FALSE)

operantChamber_basic_training <- movingAverage("../logdir/operantChamber/operantChamber-train/metrics.jsonl")

write.csv(operantChamber_basic_training, "operantChamber-training.csv", row.names = FALSE)

operantChamber_curriculum_training <- movingAverage("../../dreamer/logdir/operantChamber/operantChamber-A-train/metrics.jsonl") %>%
  bind_rows(movingAverage("../../dreamer/logdir/operantChamber/operantChamber-B-train/metrics.jsonl")) %>%
  bind_rows(movingAverage("../../dreamer/logdir/operantChamber/operantChamber-C-train/metrics.jsonl")) %>%
  bind_rows(movingAverage("../../dreamer/logdir/operantChamber/operantChamber-D-train/metrics.jsonl")) %>%
  bind_rows(movingAverage("../../dreamer/logdir/operantChamber/operantChamber-E-train/metrics.jsonl"))

write.csv(operantChamber_curriculum_training, "operantChamberCurriculum-training.csv", row.names = FALSE)

# Competition

episodes <- read.csv("../../analysis/data/competitionAAITestbed/passmarks.csv") %>%
  transmute(episode = str_remove_all(episode, ".yaml"))

episode_names_1 <- episodes %>% filter(str_detect(episode, "[0-1][0-9]-[0-3][0-9]-01")) %>% select(episode)

episode_names_2 <- episodes %>% filter(str_detect(episode, "[0-1][0-9]-[0-3][0-9]-02")) %>% select(episode)

episode_names_3 <- episodes %>% filter(str_detect(episode, "[0-1][0-9]-[0-3][0-9]-03")) %>% select(episode)


indexes <- data.frame(ind = c(1, 1, 1, 1, 1, 1,
                              2, 2, 2, 2, 2,
                              3, 3, 3),
                      ind_txt = c("ind1", "ind1_2", "ind1_3", "ind1_4", "ind1_5", "ind1_6",
                                  "ind2", "ind2_2", "ind2_3", "ind2_4", "ind2_5",
                                  "ind3", "ind3_2", "ind3_3"),
                      ind_num = c(1, 93, 160, 230, 246, 250,
                                  1, 52, 97, 168, 220,
                                  1, 238, 244))

competition_results <- data.frame(finalReward = c(), episode = c())

for (row in 1:nrow(indexes)){
  ind <- indexes$ind[row]
  path_txt <- indexes$ind_txt[row]
  ind_num <- indexes$ind_num[row]
  
  path <- paste0("../logdir/competition-curriculum/competition-curriculum-timescale300-eval-", path_txt, "/metrics.jsonl")
  
  batch <- stream_in(file(path)) %>%
    select(step, `episode/score`) %>%
    transmute(finalReward = str_remove_all(`episode/score`, "\\[|\\]"),
              finalReward = as.numeric(finalReward)) %>%
    drop_na()
  
  if (((ind_num-1) + nrow(batch)) > 300){
    delim <- 300
    episode_names <- switch(ind,
                            episode_names_1$episode[ind_num:delim],
                            episode_names_2$episode[ind_num:delim],
                            episode_names_3$episode[ind_num:delim]
    )
    
    batch <- batch %>% slice_head(n = length(episode_names)) %>%
      mutate(episode = episode_names)
    
  } else {
    delim <- (ind_num-1) + nrow(batch)
    episode_names <- switch(ind,
                            episode_names_1$episode[ind_num:delim],
                            episode_names_2$episode[ind_num:delim],
                            episode_names_3$episode[ind_num:delim]
    )
    batch <- batch %>% mutate(episode = episode_names)
  }
  competition_results <- bind_rows(competition_results, batch)
}

missing_episodes <- setdiff(select(episodes, episode), select(competition_results, episode))

first_batch_missing <- stream_in(file("../logdir/competition-curriculum/competition-curriculum-timescale300-eval-extra/metrics.jsonl")) %>%
  select(step, `episode/score`) %>%
  transmute(finalReward = str_remove_all(`episode/score`, "\\[|\\]"),
            finalReward = as.numeric(finalReward)) %>%
  drop_na() %>%
  mutate(episode = missing_episodes$episode[1:nrow(.)])

second_batch_missing <- stream_in(file("../logdir/competition-curriculum/competition-curriculum-timescale300-eval-extra2/metrics.jsonl")) %>%
  select(step, `episode/score`) %>%
  transmute(finalReward = str_remove_all(`episode/score`, "\\[|\\]"),
            finalReward = as.numeric(finalReward)) %>%
  drop_na() %>%
  slice_head(n = 5) %>%
  mutate(episode = missing_episodes$episode[30:34])

third_batch_missing <- stream_in(file("../logdir/competition-curriculum/competition-curriculum-timescale300-eval-extra3/metrics.jsonl")) %>%
  select(step, `episode/score`) %>%
  transmute(finalReward = str_remove_all(`episode/score`, "\\[|\\]"),
            finalReward = as.numeric(finalReward)) %>%
  drop_na() %>%
  slice_head(n = 3) %>%
  mutate(episode = missing_episodes$episode[27:29])


competition_results <- bind_rows(competition_results, first_batch_missing, second_batch_missing, third_batch_missing)

write.csv(competition_results, "competition-eval.csv", row.names = FALSE)
