# Load packages
library(tidyverse)
library(GGally)
library(viridis)
library(Ckmeans.1d.dp)
library(forecast)
library(ROCR)
library(caret)
library(glmnet)
library(xgboost)
library(car)
library(psych)
library(oddsratio)
library(corrplot)
library(plotmo)
library(vip)
library(rms)
library(pdp)
library(rfUtilities)
library(doSNOW)
library(doParallel)
library(fastDummies)
library(kableExtra)
library(quarto)

## create cluster
cl <- makeCluster(5) 
registerDoParallel(cl)

## Load Data ---------------------------------------------------------------
pbp_data <- read_csv("XG_Training_Data.csv") # load csv file

# Subset Data -------------------------------------------------------------
## find non shootout events
pbp_Data_clean <- pbp_data %>% 
  arrange(Season,Game_Id,Period) %>%
  filter(Player1_PositionName != "Goalie") %>%
  select(Season:Event_Team,Strength:Home_Goalie,AwayPlayer1:Player1_CurrentAge,
         Player1_Rookie,Player1_Handed,Player1_PositionName,Shot_Distance,
         Shot_Angle)

## find last events & drop NA values
last_event_df <- pbp_Data_clean %>%
  group_by(Season,Game_Id,Period) %>%
  mutate(Last_Event = lag(Event),
         Faceoff_Last = case_when(Last_Event == "FAC" ~ TRUE,
                                  Last_Event != "FAC" ~ FALSE),
         Shot_Last = case_when(Last_Event == "SHOT" ~ TRUE,
                               Last_Event != "SHOT" ~ FALSE),
         Miss_Last = case_when(Last_Event == "MISS" ~ TRUE,
                               Last_Event != "MISS" ~ FALSE),
         Block_Last = case_when(Last_Event == "BLOCK" ~ TRUE,
                                Last_Event != "BLOCK" ~ FALSE),
         Takeaway_Last = case_when(Last_Event == "TAKE" ~ TRUE,
                                   Last_Event != "TAKE" ~ FALSE),
         Giveaway_Last = case_when(Last_Event == "GIVE" ~ TRUE,
                                   Last_Event != "GIVE" ~ FALSE),
         Event_Time_Diff = abs(Seconds_Elapsed - lag(Seconds_Elapsed)),
         Shot_Distance_Last = lag(Shot_Distance),
         Shot_Angle_Last = lag(Shot_Angle),
         Shot_Distance_Diff = abs(Shot_Distance - lag(Shot_Distance)),
         Shot_Angle_Diff = abs(Shot_Angle - lag(Shot_Angle))) %>%
  filter(!is.na(Event_Time_Diff),!is.na(Shot_Distance),!is.na(Shot_Angle),
         !is.na(Shot_Distance_Last),!is.na(Shot_Angle_Last),
         !is.na(Shot_Distance_Diff),!is.na(Shot_Angle_Diff))

## find game information
game_info_df <- last_event_df %>%
  group_by(Season,Game_Id,Period) %>%
  mutate(Home = case_when(Event_Team == Home_Team ~ TRUE,
                          Event_Team == Away_Team ~ FALSE),
         Score_Diff = case_when(Home_Score == Away_Score ~ 'Tied',
                                abs(Home_Score - Away_Score) == 1 ~ '1',
                                abs(Home_Score - Away_Score) == 2 ~ '2',
                                abs(Home_Score - Away_Score) >= 3 ~ '3_Plus'))

## find rebounds 
rebounds_df <- game_info_df %>%
  group_by(Season,Game_Id,Period) %>%
  mutate(Rebound = case_when(Event_Team == lag(Event_Team) & 
                               Period == lag(Period) &
                               Last_Event == "SHOT" &
                               Event %in% c("SHOT","GOAL") &
                               ((X > 30) |
                                  X < -30) &
                               Event_Time_Diff <= 3 ~ TRUE,
                             TRUE ~ FALSE),
         Rebound_Last = case_when(Event_Team == lag(Event_Team) & 
                                    Period == lag(Period) &
                                    lag(Rebound == TRUE) ~ TRUE,
                                  TRUE ~ FALSE))

## find rush shots
rush_shots_df <- rebounds_df %>%
  group_by(Season,Game_Id,Period) %>%
  mutate(Rush_Shot = case_when(Last_Event %in% 
                                 c("SHOT","MISS","BLOCK",
                                   "FAC","GIVE","TAKE") & 
                                 Event_Team != lag(Event_Team) & 
                                 Period == lag(Period) &
                                 ((lag(X) > 30 & X < -30) | 
                                    (lag(X) < -30 & X > 30 )) &
                                 Event %in% c("SHOT","MISS","GOAL") &
                                 Event_Time_Diff <= 10 ~ TRUE, TRUE ~ FALSE),
         Rush_Shot_1 = case_when(Event_Team == lag(Event_Team) & 
                                   Period == lag(Period) &
                                   lag(Rush_Shot == TRUE) ~ TRUE,
                                 TRUE ~ FALSE),
         Rush_Shot_2 = case_when(Event_Team == lag(Event_Team) & 
                                   Period == lag(Period) &
                                   lag(Rush_Shot_1 == TRUE) ~ TRUE,
                                 TRUE ~ FALSE))

## find rush shots off a counter rush
counter_rush_df <- rush_shots_df %>%
  group_by(Season,Game_Id,Period) %>%
  mutate(Counter_Rush_Shot = case_when(Last_Event %in% 
                                         c("SHOT","MISS","BLOCK") & 
                                         Event_Team != lag(Event_Team) & 
                                         Period == lag(Period) &
                                         ((lag(X) > 30 & X < -30) | 
                                            (lag(X) < -30 & X > 30 )) &
                                         Event %in% c("SHOT","MISS","GOAL") &
                                         lag(Rush_Shot == TRUE) &
                                         Event_Time_Diff <= 12 ~ TRUE, TRUE ~ FALSE))

## find shot side
shot_side_df <- counter_rush_df %>%
  mutate(Shot_Side = case_when(((Player1_Handed == "R" & Y > 0 | 
                                   Player1_Handed == "L" & Y <= 0) & 
                                  Home == TRUE & Period %in% c(1,3,5,7) &
                                  Event %in% c("GOAL","SHOT","MISS")) | 
                                 ((Player1_Handed == "R" & Y <= 0 | 
                                     Player1_Handed == "L" & Y > 0) &
                                    Home == FALSE & Period %in% c(2,4,6,8) &
                                    Event %in% c("GOAL","SHOT","MISS")) |
                                 ((Player1_Handed == "R" & Y <= 0 | 
                                     Player1_Handed == "L" & Y > 0) & 
                                    Home == FALSE & Period %in% c(1,3,5,7) &
                                    Event %in% c("GOAL","SHOT","MISS")) | 
                                 ((Player1_Handed == "R" & Y > 0 | 
                                     Player1_Handed == "L" & Y <= 0) &
                                    Home == TRUE & Period %in% c(2,4,6,8) &
                                    Event %in% c("GOAL","SHOT","MISS")) ~ "Off_Wing",
                               ((Player1_Handed == "R" & Y <= 0 | 
                                   Player1_Handed == "L" & Y > 0) & 
                                  Home == TRUE & Period %in% c(1,3,5,7) &
                                  Event %in% c("GOAL","SHOT","MISS")) | 
                                 ((Player1_Handed == "R" & Y > 0 | 
                                     Player1_Handed == "L" & Y <= 0) &
                                    Home == FALSE & Period %in% c(2,4,6,8) &
                                    Event %in% c("GOAL","SHOT","MISS")) |
                                 ((Player1_Handed == "R" & Y > 0 | 
                                     Player1_Handed == "L" & Y <= 0) & 
                                    Home == FALSE & Period %in% c(1,3,5,7) &
                                    Event %in% c("GOAL","SHOT","MISS")) | 
                                 ((Player1_Handed == "R" & Y <= 0 | 
                                     Player1_Handed == "L" & Y > 0) &
                                    Home == TRUE & Period %in% c(2,4,6,8) &
                                    Event %in% c("GOAL","SHOT","MISS")) ~ "Strong_Side"))


## find high danger shots
danger_attempt_df <- shot_side_df %>%
  mutate(High_Danger_Attempt = case_when(((8 > Y & Y > -8) &
                                            (X > -89 & X < -59) &
                                            Event %in% c("SHOT","MISS","GOAL")) |
                                           ((8 > Y & Y > -8) &
                                              (X > 59 & X < 89) &
                                              Event %in% c("SHOT","MISS","GOAL"))
                                         ~ TRUE, TRUE ~ FALSE),
         Mediam_Danger_Attempt = case_when((((8 <= Y & Y < 17) |
                                               (-8 >= Y & Y > -17)) &
                                              (X > -89 & X < -59) &
                                              Event %in% c("SHOT","MISS","GOAL")) |
                                             (((8 <= Y & Y < 17) |
                                                 (-8 >= Y & Y > -17)) &
                                                (X > 59 & X < 89) &
                                                Event %in% c("SHOT","MISS","GOAL"))
                                           ~ TRUE, TRUE ~ FALSE),
         High_Slot_Attempt = case_when(((17 > Y & Y > -17) &
                                          (X >= -59 & X < -49) &
                                          Event %in% c("SHOT","MISS","GOAL")) |
                                         ((17 > Y & Y > -17) &
                                            (X >= 49 & X < 59) &
                                            Event %in% c("SHOT","MISS","GOAL"))
                                       ~ TRUE, TRUE ~ FALSE))

## find if last event was high danger attempt
danger_last_df <- danger_attempt_df %>%
  mutate(High_Danger_Last = case_when(lag(High_Danger_Attempt == TRUE) & 
                                        Last_Event %in% c("SHOT","MISS")
                                      ~ TRUE, TRUE ~ FALSE),
         Mediam_Danger_Last = case_when(lag(Mediam_Danger_Attempt == TRUE) & 
                                          Last_Event %in% c("SHOT","MISS")
                                        ~ TRUE, TRUE ~ FALSE),
         High_Slot_Last = case_when(lag(High_Slot_Attempt == TRUE) & 
                                      Last_Event %in% c("SHOT","MISS")
                                    ~ TRUE, TRUE ~ FALSE))


## combine common strengths
strength_df <- danger_last_df %>%
  mutate(Strength_State = case_when(Strength %in% c("5x5","4x4","3x3") ~ "Even",
                                    Strength %in% c("6x5","6x4","5x4","5x3","4x3") ~ "Power_Play",
                                    Strength %in% c("4x5","3x5","3x4","5x6","4x6") ~ "Short_Handed",
                                    TRUE ~ "Other"))


## make IS_Goal column for response variable
Final_df <- strength_df %>%
  filter(Event %in% c("GOAL","SHOT","MISS"),!is.na(Type),!is.na(Shot_Side)) %>%
  group_by(Season,Game_Id,Period) %>%
  mutate(Is_Goal = case_when(Event == "GOAL" ~ 1,
                             Event != "GOAL" ~ 0))


## specify columns to change to factor
cols <- c("Season","Period","Event_Team","Event",
          "Strength_State","Player1_Handed","Faceoff_Last","Home","Score_Diff",
          "Rebound","Rush_Shot","Shot_Side","Strength","High_Danger_Attempt",
          "High_Danger_Last","Takeaway_Last","Shot_Last","Block_Last","Miss_Last",
          "Giveaway_Last","Type","Rebound_Last","Rush_Shot_1",
          "Rush_Shot_2","Counter_Rush_Shot","Mediam_Danger_Attempt",
          "Mediam_Danger_Last","High_Slot_Attempt","High_Slot_Last")

Final_df[,cols] <- lapply(Final_df[,cols],as.factor)

# Split Data --------------------------------------------------------------
# Split Data 
## modeling data
model_df <- Final_df %>%
  filter(Season %in% c("20122013","20132014","20142015","20152016","20162017",
                       "20172018","20182019","20192020","20202021","20212022"))

## hold out data
holdout_df <- Final_df %>%
  filter(Season %in% c("20222023"))

# EDA ---------------------------------------------------------------------
## create a df for continuous variables for EDA
con_df <- model_df %>%
  ungroup() %>%
  select(Shot_Distance,Shot_Angle,Event_Time_Diff,Shot_Distance_Last,
         Shot_Angle_Last,Shot_Distance_Diff,Shot_Angle_Diff,Is_Goal)

## create a df for discrete variables for EDA
discrete_df <- model_df %>%
  select(Season,Game_Id,Period,Faceoff_Last:Giveaway_Last,Home:Is_Goal)

## summary statistics
summary(con_df)

## find standard deviation for each variable
sort(sapply(con_df, sd, na.rm = TRUE), decreasing = TRUE)

# Discrete Variables ------------------------------------------------------
## generate a list of plots
map(names(discrete_df)[1:26],
    ~ggplot(discrete_df, aes(x = !!sym(.x))) +
      geom_bar(color = "#00d100", fill="#24ff24",alpha=0.95) + labs(title = .x) +
      theme(plot.title = element_text(family="Arial", color="black", size=14, face="bold.italic"),
            axis.title.x=element_text(family="Arial", face="plain", color="black", size=14),
            axis.title.y=element_text(family="Arial", face="plain", color="black", size=14),
            axis.text.x=element_text(family="Arial", face="bold", color="black", size=8),
            axis.text.y=element_text(family="Arial", face="bold", color="black", size=8),
            panel.background=element_rect(fill="white"),
            panel.margin=unit(0.05, "lines"),
            panel.border = element_rect(color="black",fill=NA,size=1), 
            strip.background = element_rect(color="black",fill="white",size=1),
            panel.grid.major=element_blank(),
            panel.grid.minor = element_blank(),
            axis.ticks=element_blank()))

# Continuous Variables -----------------------------------------------------
## generate a list of plots
map(names(con_df)[1:8],
    ~ggplot(con_df, aes(x = !!sym(.x))) +
      geom_histogram(color = "#00d100", fill="#24ff24",alpha=0.95) + labs(title = .x) +
      theme(plot.title = element_text(family="Arial", color="black", size=14, face="bold.italic"),
            axis.title.x=element_text(family="Arial", face="plain", color="black", size=14),
            axis.title.y=element_text(family="Arial", face="plain", color="black", size=14),
            axis.text.x=element_text(family="Arial", face="bold", color="black", size=8),
            axis.text.y=element_text(family="Arial", face="bold", color="black", size=8),
            panel.background=element_rect(fill="white"),
            panel.margin=unit(0.05, "lines"),
            panel.border = element_rect(color="black",fill=NA,size=1), 
            strip.background = element_rect(color="black",fill="white",size=1),
            panel.grid.major=element_blank(),
            panel.grid.minor = element_blank(),
            axis.ticks=element_blank()))

## generate a list of plots
map(names(con_df)[1:8],
    ~ggplot(con_df, aes(x=factor(0),y = !!sym(.x))) +
      geom_boxplot(color = "#00d100", fill="#24ff24",alpha=0.95) + labs(title = .x) +
      theme(plot.title = element_text(family="Arial", color="black", size=14, face="bold.italic"),
            axis.title.y=element_text(family="Arial", face="plain", color="black", size=14),
            axis.text.y=element_text(family="Arial", face="bold", color="black", size=8),
            axis.title.x=element_blank(),
            axis.text.x=element_blank(),
            panel.background=element_rect(fill="white"),
            panel.margin=unit(0.05, "lines"),
            panel.border = element_rect(color="black",fill=NA,size=1), 
            strip.background = element_rect(color="black",fill="white",size=1),
            panel.grid.major=element_blank(),
            panel.grid.minor = element_blank(),
            axis.ticks=element_blank()))

## correlation plot
## WARNING SOME PLOTS TAKE A LONG TIME TO RENDER
## create matrix for corr plot
corr_plot_Strength <- model_df %>%
  ungroup() %>%
  select(Shot_Distance,Shot_Angle,Event_Time_Diff,Shot_Distance_Last,
         Shot_Angle_Last,Shot_Distance_Diff,Shot_Angle_Diff,Is_Goal,
         Strength_State)

## create correlation plot
## WARNING SOME PLOTS TAKE A LONG TIME TO RENDER
ggpairs(corr_plot_Strength, columns = 1:8, ggplot2::aes(color = Strength_State, alpha = 0.6),
        upper = list(continuous = wrap('cor', size = 2.5)),
        lower = list(combo = wrap("facethist", bins = 30, alpha = 0.5)),
        diag = list(continuous = wrap("densityDiag", alpha = 0.5)),
        title = "Scatterplot matrix of Shot Statistics by Strength") +
  theme(plot.title = element_text(family="Arial",color="black", size=14, 
                                  face="bold.italic"),
        axis.title.x=element_text(family="Arial", face="plain", 
                                  color="black", size=8), 
        axis.title.y=element_text(family="Arial", face="plain", 
                                  color="black", size=12), 
        axis.text.x=element_text(family="Arial", face="bold", 
                                 color="black", size=6), 
        axis.text.y=element_text(family="Arial", face="bold", 
                                 color="black", size=6), 
        panel.background=element_rect(fill="white"), 
        panel.margin=unit(0.05, "lines"), 
        panel.border = element_rect(color="black",fill=NA,size=1), 
        strip.background = element_rect(color="black",fill="white",size=1),
        panel.grid.major=element_blank(), panel.grid.minor = element_blank(), 
        axis.ticks=element_blank()) 

## create matrix for corr plot
corr_plot_type <- model_df %>%
  ungroup() %>%
  select(Shot_Distance,Shot_Angle,Event_Time_Diff,Shot_Distance_Last,
         Shot_Angle_Last,Shot_Distance_Diff,Shot_Angle_Diff,Is_Goal,Type)

## create correlation plot
## WARNING SOME PLOTS TAKE A LONG TIME TO RENDER
ggpairs(corr_plot_type, columns = 1:8, ggplot2::aes(color = Type, alpha = 0.6),
        upper = list(continuous = wrap('cor', size = 2.5)),
        lower = list(combo = wrap("facethist", bins = 30, alpha = 0.5)),
        diag = list(continuous = wrap("densityDiag", alpha = 0.5)),
        title = "Scatterplot matrix of Shot Statistics by Shot Type") +
  theme(plot.title = element_text(family="Arial",color="black", size=14, 
                                  face="bold.italic"),
        axis.title.x=element_text(family="Arial", face="plain", 
                                  color="black", size=8), 
        axis.title.y=element_text(family="Arial", face="plain", 
                                  color="black", size=12), 
        axis.text.x=element_text(family="Arial", face="bold", 
                                 color="black", size=6), 
        axis.text.y=element_text(family="Arial", face="bold", 
                                 color="black", size=6), 
        panel.background=element_rect(fill="white"), 
        panel.margin=unit(0.05, "lines"), 
        panel.border = element_rect(color="black",fill=NA,size=1), 
        strip.background = element_rect(color="black",fill="white",size=1),
        panel.grid.major=element_blank(), panel.grid.minor = element_blank(), 
        axis.ticks=element_blank()) 

## create matrix for corr plot
corr_plot_position <- model_df %>%
  ungroup() %>%
  select(Shot_Distance,Shot_Angle,Event_Time_Diff,Shot_Distance_Last,
         Shot_Angle_Last,Shot_Distance_Diff,Shot_Angle_Diff,Is_Goal,
         Player1_PositionName)

## create correlation plot
## WARNING SOME PLOTS TAKE A LONG TIME TO RENDER
ggpairs(corr_plot_position, columns = 1:4, ggplot2::aes(color = Player1_PositionName, alpha = 0.6),
        upper = list(continuous = wrap('cor', size = 2.5)),
        lower = list(combo = wrap("facethist", bins = 30, alpha = 0.5)),
        diag = list(continuous = wrap("densityDiag", alpha = 0.5)),
        title = "Scatterplot matrix of Shot Statistics by Position") +
  theme(plot.title = element_text(family="Arial",color="black", size=14, 
                                  face="bold.italic"),
        axis.title.x=element_text(family="Arial", face="plain", 
                                  color="black", size=8), 
        axis.title.y=element_text(family="Arial", face="plain", 
                                  color="black", size=12), 
        axis.text.x=element_text(family="Arial", face="bold", 
                                 color="black", size=6), 
        axis.text.y=element_text(family="Arial", face="bold", 
                                 color="black", size=6), 
        panel.background=element_rect(fill="white"), 
        panel.margin=unit(0.05, "lines"), 
        panel.border = element_rect(color="black",fill=NA,size=1), 
        strip.background = element_rect(color="black",fill="white",size=1),
        panel.grid.major=element_blank(), panel.grid.minor = element_blank(), 
        axis.ticks=element_blank()) 

## create matrix for corr plot
corr_plot_score_diff <- model_df %>%
  ungroup() %>%
  select(Shot_Distance,Shot_Angle,Event_Time_Diff,Shot_Distance_Last,
         Shot_Angle_Last,Shot_Distance_Diff,Shot_Angle_Diff,Is_Goal,Score_Diff)

## create correlation plot
## WARNING SOME PLOTS TAKE A LONG TIME TO RENDER
ggpairs(corr_plot_score_diff, columns = 1:8, ggplot2::aes(color = Score_Diff, alpha = 0.6),
        upper = list(continuous = wrap('cor', size = 2.5)),
        lower = list(combo = wrap("facethist", bins = 30, alpha = 0.5)),
        diag = list(continuous = wrap("densityDiag", alpha = 0.5)),
        title = "Scatterplot matrix of Shot Statistics by Score State") +
  theme(plot.title = element_text(family="Arial",color="black", size=14, 
                                  face="bold.italic"),
        axis.title.x=element_text(family="Arial", face="plain", 
                                  color="black", size=8), 
        axis.title.y=element_text(family="Arial", face="plain", 
                                  color="black", size=12), 
        axis.text.x=element_text(family="Arial", face="bold", 
                                 color="black", size=6), 
        axis.text.y=element_text(family="Arial", face="bold", 
                                 color="black", size=6), 
        panel.background=element_rect(fill="white"), 
        panel.margin=unit(0.05, "lines"), 
        panel.border = element_rect(color="black",fill=NA,size=1), 
        strip.background = element_rect(color="black",fill="white",size=1),
        panel.grid.major=element_blank(), panel.grid.minor = element_blank(), 
        axis.ticks=element_blank()) 

## create matrix for corr plot
corr_plot_home <- model_df %>%
  ungroup() %>%
  select(Shot_Distance,Shot_Angle,Event_Time_Diff,Shot_Distance_Last,
         Shot_Angle_Last,Shot_Distance_Diff,Shot_Angle_Diff,Is_Goal,Home)

## create correlation plot
## WARNING SOME PLOTS TAKE A LONG TIME TO RENDER
ggpairs(corr_plot_home, columns = 1:8, ggplot2::aes(color = Home, alpha = 0.6),
        upper = list(continuous = wrap('cor', size = 2.5)),
        lower = list(combo = wrap("facethist", bins = 30, alpha = 0.5)),
        diag = list(continuous = wrap("densityDiag", alpha = 0.5)),
        title = "Scatterplot matrix of Shot Statistics by Location") +
  theme(plot.title = element_text(family="Arial",color="black", size=14, 
                                  face="bold.italic"),
        axis.title.x=element_text(family="Arial", face="plain", 
                                  color="black", size=8), 
        axis.title.y=element_text(family="Arial", face="plain", 
                                  color="black", size=12), 
        axis.text.x=element_text(family="Arial", face="bold", 
                                 color="black", size=6), 
        axis.text.y=element_text(family="Arial", face="bold", 
                                 color="black", size=6), 
        panel.background=element_rect(fill="white"), 
        panel.margin=unit(0.05, "lines"), 
        panel.border = element_rect(color="black",fill=NA,size=1), 
        strip.background = element_rect(color="black",fill="white",size=1),
        panel.grid.major=element_blank(), panel.grid.minor = element_blank(), 
        axis.ticks=element_blank()) 

## create matrix for corr plot
corr_plot_shot_side <- model_df %>%
  ungroup() %>%
  select(Shot_Distance,Shot_Angle,Event_Time_Diff,Shot_Distance_Last,
         Shot_Angle_Last,Shot_Distance_Diff,Shot_Angle_Diff,Is_Goal,Shot_Side)

## create correlation plot
## WARNING SOME PLOTS TAKE A LONG TIME TO RENDER
ggpairs(corr_plot_shot_side, columns = 1:8, ggplot2::aes(color = Shot_Side, alpha = 0.6),
        upper = list(continuous = wrap('cor', size = 2.5)),
        lower = list(combo = wrap("facethist", bins = 30, alpha = 0.5)),
        diag = list(continuous = wrap("densityDiag", alpha = 0.5)),
        title = "Scatterplot matrix of Shot Statistics by Shot Side") +
  theme(plot.title = element_text(family="Arial",color="black", size=14, 
                                  face="bold.italic"),
        axis.title.x=element_text(family="Arial", face="plain", 
                                  color="black", size=8), 
        axis.title.y=element_text(family="Arial", face="plain", 
                                  color="black", size=12), 
        axis.text.x=element_text(family="Arial", face="bold", 
                                 color="black", size=6), 
        axis.text.y=element_text(family="Arial", face="bold", 
                                 color="black", size=6), 
        panel.background=element_rect(fill="white"), 
        panel.margin=unit(0.05, "lines"), 
        panel.border = element_rect(color="black",fill=NA,size=1), 
        strip.background = element_rect(color="black",fill="white",size=1),
        panel.grid.major=element_blank(), panel.grid.minor = element_blank(), 
        axis.ticks=element_blank()) 

## create matrix for corr plot
corr_plot_rebound <- model_df %>%
  ungroup() %>%
  select(Shot_Distance,Shot_Angle,Event_Time_Diff,Shot_Distance_Last,
         Shot_Angle_Last,Shot_Distance_Diff,Shot_Angle_Diff,Is_Goal,Rebound)

## create correlation plot
## WARNING SOME PLOTS TAKE A LONG TIME TO RENDER
ggpairs(corr_plot_rebound, columns = 1:8, ggplot2::aes(color = Rebound, alpha = 0.6),
        upper = list(continuous = wrap('cor', size = 2.5)),
        lower = list(combo = wrap("facethist", bins = 30, alpha = 0.5)),
        diag = list(continuous = wrap("densityDiag", alpha = 0.5)),
        title = "Scatterplot matrix of Shot Statistics by Rebound") +
  theme(plot.title = element_text(family="Arial",color="black", size=14, 
                                  face="bold.italic"),
        axis.title.x=element_text(family="Arial", face="plain", 
                                  color="black", size=8), 
        axis.title.y=element_text(family="Arial", face="plain", 
                                  color="black", size=12), 
        axis.text.x=element_text(family="Arial", face="bold", 
                                 color="black", size=6), 
        axis.text.y=element_text(family="Arial", face="bold", 
                                 color="black", size=6), 
        panel.background=element_rect(fill="white"), 
        panel.margin=unit(0.05, "lines"), 
        panel.border = element_rect(color="black",fill=NA,size=1), 
        strip.background = element_rect(color="black",fill="white",size=1),
        panel.grid.major=element_blank(), panel.grid.minor = element_blank(), 
        axis.ticks=element_blank())


## create matrix for corr plot
corr_plot_rush_shot <- model_df %>%
  ungroup() %>%
  select(Shot_Distance,Shot_Angle,Event_Time_Diff,Shot_Distance_Last,
         Shot_Angle_Last,Shot_Distance_Diff,Shot_Angle_Diff,Is_Goal,Rush_Shot)

## create correlation plot
## WARNING SOME PLOTS TAKE A LONG TIME TO RENDER
ggpairs(corr_plot_rush_shot, columns = 1:8, ggplot2::aes(color = Rush_Shot, alpha = 0.6),
        upper = list(continuous = wrap('cor', size = 2.5)),
        lower = list(combo = wrap("facethist", bins = 30, alpha = 0.5)),
        diag = list(continuous = wrap("densityDiag", alpha = 0.5)),
        title = "Scatterplot matrix of Shot Statistics by Rush Shot") +
  theme(plot.title = element_text(family="Arial",color="black", size=14, 
                                  face="bold.italic"),
        axis.title.x=element_text(family="Arial", face="plain", 
                                  color="black", size=8), 
        axis.title.y=element_text(family="Arial", face="plain", 
                                  color="black", size=12), 
        axis.text.x=element_text(family="Arial", face="bold", 
                                 color="black", size=6), 
        axis.text.y=element_text(family="Arial", face="bold", 
                                 color="black", size=6), 
        panel.background=element_rect(fill="white"), 
        panel.margin=unit(0.05, "lines"), 
        panel.border = element_rect(color="black",fill=NA,size=1), 
        strip.background = element_rect(color="black",fill="white",size=1),
        panel.grid.major=element_blank(), panel.grid.minor = element_blank(), 
        axis.ticks=element_blank()) 

## create matrix for corr plot
corr_plot_high_danger <- model_df %>%
  ungroup() %>%
  select(Shot_Distance,Shot_Angle,Event_Time_Diff,Shot_Distance_Last,
         Shot_Angle_Last,Shot_Distance_Diff,Shot_Angle_Diff,Is_Goal,
         High_Danger_Attempt)

## create correlation plot
## WARNING SOME PLOTS TAKE A LONG TIME TO RENDER
ggpairs(corr_plot_high_danger, columns = 1:8, ggplot2::aes(color = High_Danger_Attempt, alpha = 0.6),
        upper = list(continuous = wrap('cor', size = 2.5)),
        lower = list(combo = wrap("facethist", bins = 30, alpha = 0.5)),
        diag = list(continuous = wrap("densityDiag", alpha = 0.5)),
        title = "Scatterplot matrix of Shot Statistics by High Danger Attempt") +
  theme(plot.title = element_text(family="Arial",color="black", size=14, 
                                  face="bold.italic"),
        axis.title.x=element_text(family="Arial", face="plain", 
                                  color="black", size=8), 
        axis.title.y=element_text(family="Arial", face="plain", 
                                  color="black", size=12), 
        axis.text.x=element_text(family="Arial", face="bold", 
                                 color="black", size=6), 
        axis.text.y=element_text(family="Arial", face="bold", 
                                 color="black", size=6), 
        panel.background=element_rect(fill="white"), 
        panel.margin=unit(0.05, "lines"), 
        panel.border = element_rect(color="black",fill=NA,size=1), 
        strip.background = element_rect(color="black",fill="white",size=1),
        panel.grid.major=element_blank(), panel.grid.minor = element_blank(), 
        axis.ticks=element_blank()) 

## create binary dummy variables
predictors_dummy <- dummy_cols(model_df, select_columns = 
                                 c("Type",
                                   "Shot_Side",
                                   "Strength_State",
                                   "Faceoff_Last",
                                   "Takeaway_Last",
                                   "Giveaway_Last",
                                   "Miss_Last",
                                   "Block_Last",
                                   "Shot_Last",
                                   "Home","Score_Diff",
                                   "Rebound",
                                   "Rebound_Last",
                                   "Rush_Shot",
                                   "Rush_Shot_1",
                                   "Rush_Shot_2",
                                   "Counter_Rush_Shot",
                                   "High_Danger_Attempt",
                                   "High_Danger_Last",
                                   "Mediam_Danger_Attempt",
                                   "Mediam_Danger_Last",
                                   "High_Slot_Attempt",
                                   "High_Slot_Last",
                                   "Player1_PositionName"),
                               remove_first_dummy = TRUE, 
                               remove_selected_columns = TRUE)

## rename variables
predictors_dummy_clean <- predictors_dummy %>%
  rename(Defected_Shot = Type_DEFLECTED,
         Slap_Shot = `Type_SLAP SHOT`,
         Snap_Shot = `Type_SNAP SHOT`,
         Tip_In = `Type_TIP-IN`,
         Wrap_Around = `Type_WRAP-AROUND`,
         Wrist_Shot = `Type_WRIST SHOT`,
         Strong_Side = Shot_Side_Strong_Side,
         Other_Strength = Strength_State_Other,
         Power_Play = Strength_State_Power_Play,
         Short_Handed = Strength_State_Short_Handed,
         Faceoff_Last = Faceoff_Last_TRUE,
         Takeaway_Last = Takeaway_Last_TRUE,
         Giveaway_Last = Giveaway_Last_TRUE,
         Miss_Last = Miss_Last_TRUE,
         Block_Last = Block_Last_TRUE,
         Shot_Last = Shot_Last_TRUE,
         Home = Home_TRUE,
         Score_Diff_2 = Score_Diff_2,
         Score_Diff_3_Plus = Score_Diff_3_Plus,
         Score_Diff_Tied = Score_Diff_Tied,
         Rebound = Rebound_TRUE,
         Rebound_Last = Rebound_Last_TRUE,
         Rush_Shot = Rush_Shot_TRUE,
         Rush_Shot_1 = Rush_Shot_1_TRUE,
         Rush_Shot_2 = Rush_Shot_2_TRUE,
         Counter_Rush_Shot = Counter_Rush_Shot_TRUE,
         High_Danger_Attempt = High_Danger_Attempt_TRUE,
         High_Danger_Last = High_Danger_Last_TRUE,
         Mediam_Danger_Attempt = Mediam_Danger_Attempt_TRUE,
         Mediam_Danger_Last = Mediam_Danger_Last_TRUE,
         High_Slot_Attempt = High_Slot_Attempt_TRUE,
         High_Slot_Last = High_Slot_Last_TRUE,
         Defenseman = Player1_PositionName_Defenseman,
         Left_Wing = `Player1_PositionName_Left Wing`,
         Right_Wing = `Player1_PositionName_Right Wing`)

# Look at correlations between numeric features
num <- sapply(predictors_dummy_clean, FUN = is.numeric)  # identify numeric columns
(corx <- round(cor(predictors_dummy_clean[, num], use = "pairwise.complete.obs"),2))  # simple correlation matrix

# Visualize correlations; can be useful if you have a lot of features
corrplot::corrplot(corx, method = "square", order = "FPC", type = "lower", diag = TRUE,
                   cl.cex = 0.5,tl.cex = 0.3)

## subset predictors
predictors_df <- predictors_dummy_clean %>%
  select(Shot_Distance:Right_Wing)

# Redunancy analysis (but first remove response and leakage predictors)
Hmisc::redun(~., nk = 0, data = subset(predictors_df))

## convert response to factor
predictors_dummy_clean$Is_Goal <- as.numeric(predictors_dummy_clean$Is_Goal)

## rename data
model_data <- predictors_dummy_clean

## predictive analysis of expected goals data using logistic regression
## randomly split the data into training (80%) and validation (20%) datasets
set.seed(1)
train_data <- sample(rownames(model_data), nrow(model_data) * 0.8)
XG_train <- model_data[train_data, ]
test_data <- setdiff(rownames(model_data), train_data)
XG_test <- model_data[test_data, ]

## create training data for XG boost model
train <- XG_train %>%
  select(Shot_Distance:Right_Wing, -Last_Event, -Is_Goal)

## create testing data for XG boost model
test <- XG_test %>%
  select(Shot_Distance:Right_Wing, -Last_Event, - Is_Goal)

## create label for training data for XG boost model
train_label <- XG_train %>%
  select(Is_Goal)

## create label for testing data for XG boost model
test_label <- XG_test %>%
  select(Is_Goal)

## XG Boost model
dtrain <- xgb.DMatrix(data = as.matrix(train), label=train_label$Is_Goal)
dtest <- xgb.DMatrix(data = as.matrix(test), label=test_label$Is_Goal)

## watch list
watch_list <- list(train = dtrain, test = dtest)

## fit model
XG_model <- xgb.train(data = dtrain, max.depth = 6, eta = 0.005, nrounds = 5000,
                      nthread = 4, watchlist = watch_list, eval_metric = "auc",
                      eval_metric = "logloss", early_stopping_rounds = 60,
                      objective = "binary:logistic")

## error plot
XG_error <- data.frame(XG_model$evaluation_log)

## plot AUC
XG_error %>%
  ggplot(alpha=0.95) +
  geom_point(aes(XG_error$iter,XG_error$train_auc), color = "#5070D4") +
  geom_point(aes(XG_error$iter,XG_error$test_auc), color = "#DE741C") +
  ggtitle("Area Under the Curve by Iteration") +
  labs(x="Iteration", y="AUC", subtitle = "Training Data: Blue \nTesting Data: Orange") +
  theme(plot.title = element_text(family="Arial", color="black", size=14, face="bold.italic"),
        axis.title.y=element_text(family="Arial", face="plain", color="black", size=14),
        axis.text.y=element_text(family="Arial", face="bold", color="black", size=8),
        axis.title.x=element_text(family="Arial", face="plain", color="black", size=14),
        axis.text.x=element_text(family="Arial", face="bold", color="black", size=8),
        panel.background=element_rect(fill="white"),
        panel.margin=unit(0.05, "lines"),
        panel.border = element_rect(color="black",fill=NA,size=1), 
        legend.position = "right",
        legend.background = element_rect(fill = "white"),
        panel.grid.major=element_blank(),
        panel.grid.minor = element_blank(),
        axis.ticks=element_blank())

## plot AUC
XG_error %>%
  ggplot() +
  geom_point(aes(XG_error$iter,XG_error$train_logloss), color = "#5070D4") +
  geom_point(aes(XG_error$iter,XG_error$test_logloss), color = "#DE741C", alpha = 0.1) +
  ggtitle("Log Loss by Iteration") +
  labs(x="Iteration", y="Log Loss", subtitle = "Training Data: Blue \nTesting Data: Orange") +
  theme(plot.title = element_text(family="Arial", color="black", size=14, face="bold.italic"),
        axis.title.y=element_text(family="Arial", face="plain", color="black", size=14),
        axis.text.y=element_text(family="Arial", face="bold", color="black", size=8),
        axis.title.x=element_text(family="Arial", face="plain", color="black", size=14),
        axis.text.x=element_text(family="Arial", face="bold", color="black", size=8),
        panel.background=element_rect(fill="white"),
        panel.margin=unit(0.05, "lines"),
        panel.border = element_rect(color="black",fill=NA,size=1), 
        legend.position = "right",
        legend.background = element_rect(fill = "white"),
        panel.grid.major=element_blank(),
        panel.grid.minor = element_blank(),
        axis.ticks=element_blank())

## variable importance
importance <- xgb.importance(colnames(dtrain), model = XG_model)

## plot importance
xgb.ggplot.importance(importance, col = "#5070D4") +
  ggtitle("Feature Importance") +
  labs(x="Features", y="Importance") +
  theme(plot.title = element_text(family="Arial", color="black", size=14, face="bold.italic"),
        axis.title.y=element_text(family="Arial", face="plain", color="black", size=14),
        axis.text.y=element_text(family="Arial", face="bold", color="black", size=6),
        axis.title.x=element_text(family="Arial", face="plain", color="black", size=14),
        axis.text.x=element_text(family="Arial", face="bold", color="black", size=8),
        panel.background=element_rect(fill="white"),
        panel.margin=unit(0.05, "lines"),
        panel.border = element_rect(color="black",fill=NA,size=1), 
        legend.position = "right",
        legend.background = element_rect(fill = "white"),
        panel.grid.major=element_blank(),
        panel.grid.minor = element_blank(),
        axis.ticks=element_blank())

## predict
model_train_pred <- predict(XG_model, newdata = dtrain,type = "prob") 

## plot ROC curve for training data
pred_train <- prediction(model_train_pred, train_label)
perf_train <- performance(pred_train, "tpr", "fpr")
plot(perf_train, colorize=TRUE)

## predict
model_test_pred <- predict(XG_model, newdata = dtest,type = "prob")

## plot ROC curve for testing data
pred_test <- prediction(model_test_pred, test_label)
perf_test <- performance(pred_test, "tpr", "fpr")
plot(perf_test, colorize=TRUE)

## create data frame of performance metrics
Data <- c("Train","Test")

AUC <- c(round(unlist(slot(performance(pred_train, "auc"), "y.values")),4),round(unlist(slot(performance(pred_test, "auc"), "y.values")),4))

LogLoss <- c(round(min(XG_model$evaluation_log$train_logloss),4),round(min(XG_model$evaluation_log$test_logloss),4))

## show data frame of performance measures
perf_df <- data.frame(Data,AUC,LogLoss)

## show data frame of performance measures
perf_df %>%
  kbl(caption = "Best Performance Measures") %>%
  kable_classic(full_width = F, html_font = "Cambria",font_size = 10)

## partial dependency plots 
partial(XG_model, pred.var = "Shot_Distance", prob = TRUE, plot = TRUE, rug = TRUE,
        plot.engine = "ggplot2", parallel = TRUE, smooth.span = 0.9, train = train, ylab = "XG", xlab = "Shot Distance") +
  geom_rug(data = train, aes(x = Shot_Distance), alpha = 0.2, inherit.aes = FALSE) +
  ggtitle("Partial Dependency Plot") +
  labs(x="Shot Distance", y="XG") +
  theme(plot.title = element_text(family="Arial",color="black", size=14, 
                                  face="bold.italic"),
        axis.title.x=element_text(family="Arial", face="plain", 
                                  color="black", size=14), 
        axis.title.y=element_text(family="Arial", face="plain", 
                                  color="black", size=14), 
        axis.text.x=element_text(family="Arial", face="bold", 
                                 color="black", size=8), 
        axis.text.y=element_text(family="Arial", face="bold", 
                                 color="black", size=8), 
        panel.background=element_rect(fill="white"), 
        panel.margin=unit(0.05, "lines"), 
        panel.border = element_rect(color="black",fill=NA,size=1), 
        strip.background = element_rect(color="black",fill="white",size=1),
        panel.grid.major=element_blank(), panel.grid.minor = element_blank(), 
        axis.ticks=element_blank())

## partial dependency plots 
partial(XG_model, pred.var = "Shot_Angle", prob = TRUE, plot = TRUE, rug = TRUE,
        plot.engine = "ggplot2", parallel = TRUE, smooth.span = 0.9, train = train) +
  geom_rug(data = train, aes(x = Shot_Angle), alpha = 0.2, inherit.aes = FALSE) +
  ggtitle("Partial Dependency Plot") +
  labs(x="Shot Angle", y="XG") +
  theme(plot.title = element_text(family="Arial",color="black", size=14, 
                                  face="bold.italic"),
        axis.title.x=element_text(family="Arial", face="plain", 
                                  color="black", size=14), 
        axis.title.y=element_text(family="Arial", face="plain", 
                                  color="black", size=14), 
        axis.text.x=element_text(family="Arial", face="bold", 
                                 color="black", size=8), 
        axis.text.y=element_text(family="Arial", face="bold", 
                                 color="black", size=8), 
        panel.background=element_rect(fill="white"), 
        panel.margin=unit(0.05, "lines"), 
        panel.border = element_rect(color="black",fill=NA,size=1), 
        strip.background = element_rect(color="black",fill="white",size=1),
        panel.grid.major=element_blank(), panel.grid.minor = element_blank(), 
        axis.ticks=element_blank())

## partial dependency plots 
partial(XG_model, pred.var = "Shot_Distance_Last", prob = TRUE, plot = TRUE, rug = TRUE,
        plot.engine = "ggplot2", parallel = TRUE, smooth.span = 0.9, train = train) +
  geom_rug(data = train, aes(x = Shot_Distance_Last), alpha = 0.2, inherit.aes = FALSE) +
  ggtitle("Partial Dependency Plot") +
  labs(x="Last Shot Distance", y="XG") +
  theme(plot.title = element_text(family="Arial",color="black", size=14, 
                                  face="bold.italic"),
        axis.title.x=element_text(family="Arial", face="plain", 
                                  color="black", size=14), 
        axis.title.y=element_text(family="Arial", face="plain", 
                                  color="black", size=14), 
        axis.text.x=element_text(family="Arial", face="bold", 
                                 color="black", size=8), 
        axis.text.y=element_text(family="Arial", face="bold", 
                                 color="black", size=8), 
        panel.background=element_rect(fill="white"), 
        panel.margin=unit(0.05, "lines"), 
        panel.border = element_rect(color="black",fill=NA,size=1), 
        strip.background = element_rect(color="black",fill="white",size=1),
        panel.grid.major=element_blank(), panel.grid.minor = element_blank(), 
        axis.ticks=element_blank())

## partial dependency plots 
partial(XG_model, pred.var = "Shot_Angle_Last", prob = TRUE, plot = TRUE, rug = TRUE,
        plot.engine = "ggplot2", parallel = TRUE, smooth.span = 0.9, train = train) +
  geom_rug(data = train, aes(x = Shot_Angle_Last), alpha = 0.2, inherit.aes = FALSE) +
  ggtitle("Partial Dependency Plot") +
  labs(x="Last Shot Angle", y="XG") +
  theme(plot.title = element_text(family="Arial",color="black", size=14, 
                                  face="bold.italic"),
        axis.title.x=element_text(family="Arial", face="plain", 
                                  color="black", size=14), 
        axis.title.y=element_text(family="Arial", face="plain", 
                                  color="black", size=14), 
        axis.text.x=element_text(family="Arial", face="bold", 
                                 color="black", size=8), 
        axis.text.y=element_text(family="Arial", face="bold", 
                                 color="black", size=8), 
        panel.background=element_rect(fill="white"), 
        panel.margin=unit(0.05, "lines"), 
        panel.border = element_rect(color="black",fill=NA,size=1), 
        strip.background = element_rect(color="black",fill="white",size=1),
        panel.grid.major=element_blank(), panel.grid.minor = element_blank(), 
        axis.ticks=element_blank())

## partial dependency plots 
partial(XG_model, pred.var = "Event_Time_Diff", prob = TRUE, plot = TRUE, rug = TRUE,
        plot.engine = "ggplot2", parallel = TRUE, smooth.span = 0.9, train = train) +
  geom_rug(data = train, aes(x = Event_Time_Diff), alpha = 0.2, inherit.aes = FALSE) +
  ggtitle("Partial Dependency Plot") +
  labs(x="Time Between Events", y="XG") +
  theme(plot.title = element_text(family="Arial",color="black", size=14, 
                                  face="bold.italic"),
        axis.title.x=element_text(family="Arial", face="plain", 
                                  color="black", size=14), 
        axis.title.y=element_text(family="Arial", face="plain", 
                                  color="black", size=14), 
        axis.text.x=element_text(family="Arial", face="bold", 
                                 color="black", size=8), 
        axis.text.y=element_text(family="Arial", face="bold", 
                                 color="black", size=8), 
        panel.background=element_rect(fill="white"), 
        panel.margin=unit(0.05, "lines"), 
        panel.border = element_rect(color="black",fill=NA,size=1), 
        strip.background = element_rect(color="black",fill="white",size=1),
        panel.grid.major=element_blank(), panel.grid.minor = element_blank(), 
        axis.ticks=element_blank())

## partial dependency plots 
partial(XG_model, pred.var = "Rebound", prob = TRUE, plot = TRUE, rug = TRUE,
        plot.engine = "ggplot2", parallel = TRUE, smooth.span = 0.9, train = train) +
  geom_rug(data = train, aes(x = Rebound), alpha = 0.2, inherit.aes = FALSE) +
  ggtitle("Partial Dependency Plot") +
  labs(x="Rebound", y="XG") +
  theme(plot.title = element_text(family="Arial",color="black", size=14, 
                                  face="bold.italic"),
        axis.title.x=element_text(family="Arial", face="plain", 
                                  color="black", size=14), 
        axis.title.y=element_text(family="Arial", face="plain", 
                                  color="black", size=14), 
        axis.text.x=element_text(family="Arial", face="bold", 
                                 color="black", size=8), 
        axis.text.y=element_text(family="Arial", face="bold", 
                                 color="black", size=8), 
        panel.background=element_rect(fill="white"), 
        panel.margin=unit(0.05, "lines"), 
        panel.border = element_rect(color="black",fill=NA,size=1), 
        strip.background = element_rect(color="black",fill="white",size=1),
        panel.grid.major=element_blank(), panel.grid.minor = element_blank(), 
        axis.ticks=element_blank())

## partial dependency plots 
partial(XG_model, pred.var = "High_Danger_Attempt", prob = TRUE, plot = TRUE, rug = TRUE,
        plot.engine = "ggplot2", parallel = TRUE, smooth.span = 0.9, train = train) +
  geom_rug(data = train, aes(x = High_Danger_Attempt), alpha = 0.2, inherit.aes = FALSE) +
  ggtitle("Partial Dependency Plot") +
  labs(x="High Danger Attempt", y="XG") +
  theme(plot.title = element_text(family="Arial",color="black", size=14, 
                                  face="bold.italic"),
        axis.title.x=element_text(family="Arial", face="plain", 
                                  color="black", size=14), 
        axis.title.y=element_text(family="Arial", face="plain", 
                                  color="black", size=14), 
        axis.text.x=element_text(family="Arial", face="bold", 
                                 color="black", size=8), 
        axis.text.y=element_text(family="Arial", face="bold", 
                                 color="black", size=8), 
        panel.background=element_rect(fill="white"), 
        panel.margin=unit(0.05, "lines"), 
        panel.border = element_rect(color="black",fill=NA,size=1), 
        strip.background = element_rect(color="black",fill="white",size=1),
        panel.grid.major=element_blank(), panel.grid.minor = element_blank(), 
        axis.ticks=element_blank())

## calibration curve
y01 <- ifelse(test == "1", 1, 0)

rms::val.prob(model_test_pred, y = y01)

## create binary dummy variables
holdout_predictors_dummy <- dummy_cols(holdout_df, select_columns = 
                                         c("Type",
                                           "Shot_Side",
                                           "Strength_State",
                                           "Faceoff_Last",
                                           "Takeaway_Last",
                                           "Giveaway_Last",
                                           "Miss_Last",
                                           "Block_Last",
                                           "Shot_Last",
                                           "Home","Score_Diff",
                                           "Rebound",
                                           "Rebound_Last",
                                           "Rush_Shot",
                                           "Rush_Shot_1",
                                           "Rush_Shot_2",
                                           "Counter_Rush_Shot",
                                           "High_Danger_Attempt",
                                           "High_Danger_Last",
                                           "Mediam_Danger_Attempt",
                                           "Mediam_Danger_Last",
                                           "High_Slot_Attempt",
                                           "High_Slot_Last",
                                           "Player1_PositionName"),
                                       remove_first_dummy = TRUE, 
                                       remove_selected_columns = TRUE)

glimpse(holdout_predictors_dummy)

## rename variables
holdout_dummy <- holdout_predictors_dummy %>%
  rename(Defected_Shot = Type_DEFLECTED,
         Slap_Shot = `Type_SLAP SHOT`,
         Snap_Shot = `Type_SNAP SHOT`,
         Tip_In = `Type_TIP-IN`,
         Wrap_Around = `Type_WRAP-AROUND`,
         Wrist_Shot = `Type_WRIST SHOT`,
         Strong_Side = Shot_Side_Strong_Side,
         Other_Strength = Strength_State_Other,
         Power_Play = Strength_State_Power_Play,
         Short_Handed = Strength_State_Short_Handed,
         Faceoff_Last = Faceoff_Last_TRUE,
         Takeaway_Last = Takeaway_Last_TRUE,
         Giveaway_Last = Giveaway_Last_TRUE,
         Miss_Last = Miss_Last_TRUE,
         Block_Last = Block_Last_TRUE,
         Shot_Last = Shot_Last_TRUE,
         Home = Home_TRUE,
         Score_Diff_2 = Score_Diff_2,
         Score_Diff_3_Plus = Score_Diff_3_Plus,
         Score_Diff_Tied = Score_Diff_Tied,
         Rebound = Rebound_TRUE,
         Rebound_Last = Rebound_Last_TRUE,
         Rush_Shot = Rush_Shot_TRUE,
         Rush_Shot_1 = Rush_Shot_1_TRUE,
         Rush_Shot_2 = Rush_Shot_2_TRUE,
         Counter_Rush_Shot = Counter_Rush_Shot_TRUE,
         High_Danger_Attempt = High_Danger_Attempt_TRUE,
         High_Danger_Last = High_Danger_Last_TRUE,
         Mediam_Danger_Attempt = Mediam_Danger_Attempt_TRUE,
         Mediam_Danger_Last = Mediam_Danger_Last_TRUE,
         High_Slot_Attempt = High_Slot_Attempt_TRUE,
         High_Slot_Last = High_Slot_Last_TRUE,
         Defenseman = Player1_PositionName_Defenseman,
         Left_Wing = `Player1_PositionName_Left Wing`,
         Right_Wing = `Player1_PositionName_Right Wing`)

glimpse(holdout_dummy)

## split data between predictors & non-predictors
holdout_data <- holdout_dummy %>%
  select(Shot_Distance:Right_Wing, -Last_Event, -Is_Goal)

## split data between predictors & non-predictors
holdout_info <- holdout_df %>%
  select(Season:Shot_Distance,Last_Event,Is_Goal)

## create label for holdout data for prediction
holdout_label <- holdout_dummy %>%
  select(Is_Goal)

## create DMatrix
dhold_out <- xgb.DMatrix(data = as.matrix(holdout_data), label=holdout_label$Is_Goal)

## convert response to factor
holdout_predictors_dummy_clean$Is_Goal <- as.factor(holdout_predictors_dummy_clean$Is_Goal)

## create df for predictors
holdout_df <- holdout_predictors_dummy_clean %>%
  select(Shot_Distance:Right_Wing, -Last_Event, -Is_Goal)

## create label for data for XG boost model
holdout_label <- holdout_predictors_dummy_clean %>%
  select(Is_Goal)

## create df for player info
holdout_info <- holdout_predictors_dummy_clean %>%
  select(Season:Player1_Handed,Last_Event)

## predict
model_holdout_pred <- predict(XG_model, newdata = dhold_out,type = "prob") 

## plot ROC curve for training data
pred_holdout <- prediction(model_holdout_pred, holdout_label)
perf_holdout <- performance(pred_holdout, "tpr", "fpr")
plot(perf_holdout, colorize=TRUE)

## create data frame of performance metrics
Data <- c("Holdout")

AUC <- c(round(unlist(slot(performance(pred_holdout, "auc"), "y.values")),4))

LogLoss <- c(round(logLoss(holdout_label$Is_Goal,model_holdout_pred),4))

## show data frame of performance measures
perf_df <- data.frame(Data,AUC,LogLoss)

## show data frame of performance measures
perf_df %>%
  kbl(caption = "Holdout Performance Measures") %>%
  kable_classic(full_width = F, html_font = "Cambria",font_size = 10)

## convert predictions to data frame
pred_2023 <- as.data.frame(model_holdout_pred)

## join data with prob
XG_df <- cbind(holdout_df,pred_2023)

## rename data
XG_final <- XG_df%>%
  rename(XG = model_holdout_pred) 

## round XG
XG_final$XG <- round((XG_final$XG),4)

## subset data
XG_players_values <- XG_final %>%
  ungroup() %>%
  group_by(Player1_FullName,Event_Team) %>%
  summarise(Total_XG = sum(XG),
            Total_Actual_Goals = sum(Is_Goal)) %>%
  mutate(XG_Difference = abs(Total_Actual_Goals - Total_XG)) %>%
  dplyr::select(Event_Team,Total_XG,Player1_FullName,
                Total_Actual_Goals,Total_XG,XG_Difference) %>%
  arrange(desc(Total_XG))

XG_players_values %>%
  head(25) %>%
  kbl(caption = "Top 25 Expected Goals Leaders as of January") %>%
  kable_classic(full_width = F, html_font = "Cambria",font_size = 10)

## plots
XG_final %>% 
  ggplot(aes(x=Type,y=XG,fill=Type)) + 
  geom_violin(alpha=0.95) +
  geom_boxplot(width=0.05, fill="black", outlier.color = NA) + 
  stat_summary(fun = median,geom="point",fill="white",size=2.5,shape=21) +
  ggtitle("Expected Goals by Shot Type & Strength") +
  labs(x="Shot Type", y="XG", fill="Shot Type",
       caption = "Data source: NHL API") +
  theme(plot.title = element_text(family="Arial",color="black",size=14,face="bold.italic"),
        axis.title.y=element_text(family="Arial",face="plain",color="black",size=14),
        legend.title = element_text(family="Arial",face="bold",colour="black",size=10),
        axis.text.y=element_text(family="Arial",face="bold",color="black",size=8),
        panel.margin=unit(0.05, "lines"),
        panel.border = element_rect(color="black",fill=NA,size=1), 
        strip.background = element_rect(color="black",fill="white",size=1),
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        panel.background=element_blank(),
        panel.grid.major=element_blank(),
        panel.grid.minor = element_blank(),
        axis.ticks=element_blank(),
        legend.position = "right",
        legend.background = element_rect(fill = "white")) +
  facet_wrap(~Strength_State)

## plots
XG_final %>% 
  ggplot(aes(x=Type,y=XG,fill=Type)) + 
  geom_violin(alpha=0.95) +
  geom_boxplot(width=0.05, fill="black", outlier.color = NA) + 
  stat_summary(fun = median,geom="point",fill="white",size=2.5,shape=21) +
  ggtitle("Expected Goals by Shot Type & Rebounds") +
  labs(x="Shot Type", y="XG", fill="Shot Type",
       caption = "Data source: NHL API") +
  theme(plot.title = element_text(family="Arial",color="black",size=14,face="bold.italic"),
        axis.title.y=element_text(family="Arial",face="plain",color="black",size=14),
        legend.title = element_text(family="Arial",face="bold",colour="black",size=10),
        axis.text.y=element_text(family="Arial",face="bold",color="black",size=8),
        panel.margin=unit(0.05, "lines"),
        panel.border = element_rect(color="black",fill=NA,size=1), 
        strip.background = element_rect(color="black",fill="white",size=1),
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        panel.background=element_blank(),
        panel.grid.major=element_blank(),
        panel.grid.minor = element_blank(),
        axis.ticks=element_blank(),
        legend.position = "right",
        legend.background = element_rect(fill = "white")) +
  facet_wrap(~Rebound)

## plots
XG_final %>% 
  ggplot(aes(x=Type,y=XG,fill=Type)) + 
  geom_violin(alpha=0.95) +
  geom_boxplot(width=0.05, fill="black", outlier.color = NA) + 
  stat_summary(fun = median,geom="point",fill="white",size=2.5,shape=21) +
  ggtitle("Expected Goals by Shot Type & Rush Shots") +
  labs(x="Shot Type", y="XG", fill="Shot Type",
       caption = "Data source: NHL API") +
  theme(plot.title = element_text(family="Arial",color="black",size=14,face="bold.italic"),
        axis.title.y=element_text(family="Arial",face="plain",color="black",size=14),
        legend.title = element_text(family="Arial",face="bold",colour="black",size=10),
        axis.text.y=element_text(family="Arial",face="bold",color="black",size=8),
        panel.margin=unit(0.05, "lines"),
        panel.border = element_rect(color="black",fill=NA,size=1), 
        strip.background = element_rect(color="black",fill="white",size=1),
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        panel.background=element_blank(),
        panel.grid.major=element_blank(),
        panel.grid.minor = element_blank(),
        axis.ticks=element_blank(),
        legend.position = "right",
        legend.background = element_rect(fill = "white")) +
  facet_wrap(~Rush_Shot)

## plots
XG_final %>% 
  ggplot(aes(x=Type,y=XG,fill=Type)) + 
  geom_violin(alpha=0.95) +
  geom_boxplot(width=0.05, fill="black", outlier.color = NA) + 
  stat_summary(fun = median,geom="point",fill="white",size=2.5,shape=21) +
  ggtitle("Expected Goals by Shot Type & High Danger Attempts") +
  labs(x="Shot Type", y="XG", fill="Shot Type",
       caption = "Data source: NHL API") +
  theme(plot.title = element_text(family="Arial",color="black",size=14,face="bold.italic"),
        axis.title.y=element_text(family="Arial",face="plain",color="black",size=14),
        legend.title = element_text(family="Arial",face="bold",colour="black",size=10),
        axis.text.y=element_text(family="Arial",face="bold",color="black",size=8),
        panel.margin=unit(0.05, "lines"),
        panel.border = element_rect(color="black",fill=NA,size=1), 
        strip.background = element_rect(color="black",fill="white",size=1),
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        panel.background=element_blank(),
        panel.grid.major=element_blank(),
        panel.grid.minor = element_blank(),
        axis.ticks=element_blank(),
        legend.position = "right",
        legend.background = element_rect(fill = "white")) +
  facet_wrap(~High_Danger_Attempt)

## plots
XG_final %>% 
  ggplot(aes(x=Type,y=XG,fill=Type)) + 
  geom_violin(alpha=0.95) +
  geom_boxplot(width=0.05, fill="black", outlier.color = NA) + 
  stat_summary(fun = median,geom="point",fill="white",size=2.5,shape=21) +
  ggtitle("Expected Goals by Shot Type & Score Difference") +
  labs(x="Shot Type", y="XG", fill="Shot Type",
       caption = "Data source: NHL API") +
  theme(plot.title = element_text(family="Arial",color="black",size=14,face="bold.italic"),
        axis.title.y=element_text(family="Arial",face="plain",color="black",size=14),
        legend.title = element_text(family="Arial",face="bold",colour="black",size=10),
        axis.text.y=element_text(family="Arial",face="bold",color="black",size=8),
        panel.margin=unit(0.05, "lines"),
        panel.border = element_rect(color="black",fill=NA,size=1), 
        strip.background = element_rect(color="black",fill="white",size=1),
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        panel.background=element_blank(),
        panel.grid.major=element_blank(),
        panel.grid.minor = element_blank(),
        axis.ticks=element_blank(),
        legend.position = "right",
        legend.background = element_rect(fill = "white")) +
  facet_wrap(~Score_Diff)

## plots
XG_final %>% 
  ggplot(aes(x=Type,y=XG,fill=Type)) + 
  geom_violin(alpha=0.95) +
  geom_boxplot(width=0.05, fill="black", outlier.color = NA) + 
  stat_summary(fun = median,geom="point",fill="white",size=2.5,shape=21) +
  ggtitle("Expected Goals by Shot Type & Home Ice") +
  labs(x="Shot Type", y="XG", fill="Shot Type",
       caption = "Data source: NHL API") +
  theme(plot.title = element_text(family="Arial",color="black",size=14,face="bold.italic"),
        axis.title.y=element_text(family="Arial",face="plain",color="black",size=14),
        legend.title = element_text(family="Arial",face="bold",colour="black",size=10),
        axis.text.y=element_text(family="Arial",face="bold",color="black",size=8),
        panel.margin=unit(0.05, "lines"),
        panel.border = element_rect(color="black",fill=NA,size=1), 
        strip.background = element_rect(color="black",fill="white",size=1),
        axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        panel.background=element_blank(),
        panel.grid.major=element_blank(),
        panel.grid.minor = element_blank(),
        axis.ticks=element_blank(),
        legend.position = "right",
        legend.background = element_rect(fill = "white")) +
  facet_wrap(~Home)

## plots
XG_final %>% 
  ggplot(aes(x=Shot_Distance,y=XG)) +                 
  geom_point(color="blue",alpha=0.1) + 
  scale_x_log10(n.breaks=4,limits = c(5, 100),
                breaks=c(5,10,25,50,100)) +
  stat_density_2d(aes(fill=stat(level)),geom="polygon",bins=75) +
  facet_wrap(~Strength_State) +   
  scale_fill_viridis(option = "A") +
  ggtitle("Density Contour of Shot Distance & Expected Goals by Strength") +
  labs(x="Log Shot Distance", y="XG", fill="Level",
       caption = "Data source: NHL API") +
  theme(plot.title = element_text(family="Arial",color="black",size=14,face="bold.italic"),
        axis.title.x=element_text(family="Arial",face="plain",color="black",size=14),
        axis.title.y=element_text(family="Arial",face="plain",color="black",size=14),
        legend.title = element_text(family="Arial",face="bold",color="black",size=10),
        axis.text.x=element_text(family="Arial",face="bold",color="black",size=8),
        axis.text.y=element_text(family="Arial",face="bold",color="black",size=8),
        panel.background=element_rect(fill = "White"),
        panel.margin=unit(0.05, "lines"),
        panel.border = element_rect(color="black",fill=NA,size = 1), 
        strip.background = element_rect(color="black",fill="white",size=1),
        panel.grid.major=element_blank(),
        panel.grid.minor = element_blank(),
        axis.ticks=element_blank())

## plots
XG_final %>% 
  ggplot(aes(x=Event_Time_Diff,y=XG)) +                 
  geom_point(color="blue",alpha=0.1) + 
  scale_x_continuous(n.breaks=4,limits = c(0, 100),
                     breaks=c(0,25,50,75,100)) +
  stat_density_2d(aes(fill=stat(level)),geom="polygon",bins=75) +
  facet_wrap(~Strength_State) +   
  scale_fill_viridis(option = "A") +
  ggtitle("Density Contour of Time Difference & Expected Goals by Strength") +
  labs(x="Time Difference", y="XG", fill="Level",
       caption = "Data source: NHL API") +
  theme(plot.title = element_text(family="Arial",color="black",size=14,face="bold.italic"),
        axis.title.x=element_text(family="Arial",face="plain",color="black",size=14),
        axis.title.y=element_text(family="Arial",face="plain",color="black",size=14),
        legend.title = element_text(family="Arial",face="bold",color="black",size=10),
        axis.text.x=element_text(family="Arial",face="bold",color="black",size=8),
        axis.text.y=element_text(family="Arial",face="bold",color="black",size=8),
        panel.background=element_rect(fill = "White"),
        panel.margin=unit(0.05, "lines"),
        panel.border = element_rect(color="black",fill=NA,size = 1), 
        strip.background = element_rect(color="black",fill="white",size=1),
        panel.grid.major=element_blank(),
        panel.grid.minor = element_blank(),
        axis.ticks=element_blank())