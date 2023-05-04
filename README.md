# Hockey-XG-Model
A new NHL expected goals model using extreme gradient boosting. 

AUTHOR
Nick Glass

Overview:

The goal of this project was to create a model to predict the probability of shot becoming a goal for NHL players. I think there is value in better understanding how scoring chances change based on different factors in a hockey game. Anyone who is a hockey fan would say that taking a shot right in front of the goal would be a better scoring chance than shooting the puck from the corner or neutral zone. But how much better is that “grade A” scoring chance? What if the player was taking a wrist shot or a slap shot? What if it the shot was off a rebound or on the power play? This model will provide context to all of these unique scenarios. In the words of the famous statistician George Box, “All models are wrong, some are useful.” The overall objective of this project was to be as useful as possible in providing context to the sport we love.

Methodology:

The method used for this task was extreme gradient boosting with a simple logistic regression model as a baseline model. Other methods induced elastic-net regression and random forests but in the end the best model was the extreme gradient boosting model. I will only focus on the final model in this report in order to be as concise as possible. New variables were created from the NHL play-by-play data in order to give a more detailed analysis. The data for this project came from Harry Shomer’s python package and the NHLapi package. The data was joined in a separate script and imported for this project.

Abstract:

This model was created in three main steps. The first step included the creation, cleaning, and examination of new variables. These variables are shown in the dictionary. The second step included the model preparation and execution. The final main step showed the model performance, interpretation, and findings.

The performance of this model was measured in terms of log loss as the primary metric since we are looking for the probability of scoring not classifying events as a binary outcome. With this said I did record the area under the curve to help better understand the models performance. The area under the curve for the training data was 0.8336 and the log loss was 0.1878. For the testing data the area under the curve was 0.8205 and the log loss was 0.1918. When predicting the probability for the 2022-23 season these values were slightly worse with an AUC of 0.7951 and a log loss of 0.2109 but that is understandable because the model did not see this data prior. This unseen data was called the holdout sample.

Pitfalls:

Potential pitfalls of this model include the lack of player and puck tracking data, as well as goalie information. We will not be able to capture the effects of player location without the puck or how a pass effects the scoring chance. Furthermore, goalie size and other information was absent from the model and might be worth inducing in future projects. Another possible issue that will need to be addressed in my future work is the strength states during the game. There were a fair amount of observations that were abnormal in terms of number of skaters on the ice. I did not want to loose information or create a large number of categories so I binned these values into a single value called “other”. There are probably better ways to handle this issue but for this iteration of the model I decided to handle this issue that way. The last potential issue I wanted to acknowledge was I included empty net goals and non-empty net goals together in this model. This could bias the model since it would treat both goals the same in terms of probability. I will fix the issue in the near future as it is important to capture this aspect of the game.

Variable Dictionary:

Faceoff Last- Was the previous event a faceoff? (TRUE or FALSE).

Shot Last- Was the previous event a shot? (TRUE or FALSE).

Missed Last- Was the previous event a Missed shot? (TRUE or FALSE).

Block Last- Was the previous event a blocked shot? (TRUE or FALSE).

Takeaway Last- Was the previous event a takeaway? (TRUE or FALSE).

Giveaway Last- Was the previous event a giveaway? (TRUE or FALSE).

Event Time Difference - The time between events in seconds.

X - The x coordinate corresponding to the location of the event on the rink from left to right with values of -99 to 99 in ft (Rink is 200ft long with 11ft behind each net).

Y - The fixed y coordinate corresponding to the location of the event on the rink from bottom to top with values of -42 to 42 in ft (Rink is 85ft wide).

Shot Distance - The distance in feet of the shot from the net.

Shot Angle - The angle of the shot from the net.

Shot Distance Diff - The difference in the shot distance from the last shot location.

Shot Angle Diff - The difference in the shot angle from the last shot location.

Shot Distance Last - The last shot distance.

Shot Angle Last - The difference in the shot angle from the last shot location.

Home - Is the event team the home team? (TRUE or FALSE).

Score Diff - What is the difference in the games score? (Tied, 1, 2, 3+).

Rebound - Is the play a rebound? Defined as a shot on goal followed by a shot on goal or a goal by the same event team in the same period, within 3 seconds of each other (TRUE or FALSE).

Rebound Last - Is the last play a rebound? (TRUE or FALSE).

Rush Shot - Is the shot attempt off the rush? Defined as a shot on goal, a missed shot, or a goal, that occurred within 10 seconds of the opposing teams last shot attempt. (TRUE or FALSE).

Rush Shot 1 - Is the last play a rush shot? (TRUE or FALSE).

Rush Shot 2 - Is the second to last play a rush shot? (TRUE or FALSE).

Counter Rush Shot - Is the shot off another teams failed rush attempt? (TRUE or FALSE).

Shot Side - Is the player on their strong side or off wing when taking the shot?

High Danger Attempt - Did the shot attempt occur in the slot? Defined as 30ft in front of the crease and 16ft wide between the faceoff circles. (TRUE or FALSE).

High Danger Last - Was the last event a high danger attempt? (TRUE or FALSE).

Medium Danger Attempt - Did the shot attempt occur next to the slot? (TRUE or FALSE).

Medium Danger Last - Was the last event a medium danger attempt? (TRUE or FALSE).

High Slot Attempt - Did the shot attempt occur above the slot? (TRUE or FALSE).

High Slot Last - Was the last event a high slot attempt? (TRUE or FALSE).

Strength State - The strength of the event that occurred. (Even, Power Play, Short Handed, etc.)

Type - The type of shot that occurred.

Player 1 Position Name - The position of the player taking the shot. (Excluding Goalies)

Is Goal - Did the event result in a goal? (This is the response variable, listed as 0 or 1).

Rink Diagram:

![image](https://user-images.githubusercontent.com/113626253/236293271-da5e59ee-98c3-4aed-a550-2cea4923c69d.png)

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

## Load Data
pbp_data <- read_csv("pbp_final.csv") # load csv file

## subset data
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

## make column for response variable
Final_df <- strength_df %>%
  filter(Event %in% c("GOAL","SHOT","MISS"),!is.na(Type),!is.na(Shot_Side)) %>%
  group_by(Season,Game_Id,Period) %>%
  mutate(Is_Goal = case_when(Event == "GOAL" ~ 1,
                             Event != "GOAL" ~ 0))

glimpse(Final_df)

These features were created using existing data and knowledge of the sport. The times used as a threshold for rebounds and rush shots were picked based on my knowledge of the game. I debated making the time for a rush shot to occur much less but decided to keep it at 10 seconds to have a larger sample size of rush shots. The same method was used to define the area for different scoring chances. It would be interesting in future work to see how the model changes based on engineering features in alternative ways.

## specify columns to change to factor
cols <- c("Season","Period","Event_Team","Event",
          "Strength_State","Player1_Handed","Faceoff_Last","Home","Score_Diff",
          "Rebound","Rush_Shot","Shot_Side","Strength","High_Danger_Attempt",
          "High_Danger_Last","Takeaway_Last","Shot_Last","Block_Last","Miss_Last",
          "Giveaway_Last","Type","Rebound_Last","Rush_Shot_1",
          "Rush_Shot_2","Counter_Rush_Shot","Mediam_Danger_Attempt",
          "Mediam_Danger_Last","High_Slot_Attempt","High_Slot_Last")

Final_df[,cols] <- lapply(Final_df[,cols],as.factor)

# Split Data 
## modeling data
model_df <- Final_df %>%
  filter(Season %in% c("20122013","20132014","20142015","20152016","20162017",
                       "20172018","20182019","20192020","20202021","20212022"))

## hold out data
holdout_df <- Final_df %>%
  filter(Season %in% c("20222023"))

**Summary Statistics:**

![image](https://user-images.githubusercontent.com/113626253/236294759-12ad6c91-d62e-4650-be1d-4a62d0a7e01c.png)

![image](https://user-images.githubusercontent.com/113626253/236294858-279bf8e2-002d-414d-b67f-c75862ed0154.png)

A few takeaways from the summary statistics were the fact that the time between events is likely skewed due to having such a large maximum time. We could remove these outliers but that could causes a loss of information. Another interesting note is that the there was fairly large standard deviation for the last shot angle and distance. This indicates that the data was very spread out for these variables.

![image](https://user-images.githubusercontent.com/113626253/236295031-056574c3-e4e7-4b0a-8d15-d4b3046e3e15.png)


![image](https://user-images.githubusercontent.com/113626253/236295073-56e23f8b-f6b2-4b29-ae6f-202446b5eadf.png)


![image](https://user-images.githubusercontent.com/113626253/236295113-e5b18aad-46b1-4b4e-b0af-02766be611aa.png)


![image](https://user-images.githubusercontent.com/113626253/236295142-942c7d54-2712-4fbe-a020-4f95d93892ab.png)


![image](https://user-images.githubusercontent.com/113626253/236295191-597d9946-e7bb-4ad5-a3b6-c032527380c9.png)


![image](https://user-images.githubusercontent.com/113626253/236295246-15c3f704-ef7a-4c42-949e-68d560406096.png)


![image](https://user-images.githubusercontent.com/113626253/236295287-77580c28-56be-4742-810c-815f9e9c9c63.png)


![image](https://user-images.githubusercontent.com/113626253/236295324-ad42b94d-5d30-40a8-aa65-6fcebea5d48c.png)


![image](https://user-images.githubusercontent.com/113626253/236295353-96983fe7-b211-4c2d-89c2-21c6822543f2.png)


![image](https://user-images.githubusercontent.com/113626253/236295391-0377aa0f-e2b6-487f-a4ed-aeeb911b6366.png)


![image](https://user-images.githubusercontent.com/113626253/236295478-44892639-e682-44de-b0bf-f2dbf86c9593.png)


![image](https://user-images.githubusercontent.com/113626253/236295527-16ef7651-17e5-4973-83d9-480d56e5f61d.png)


![image](https://user-images.githubusercontent.com/113626253/236295556-55bf4d08-89c0-4d9c-9525-dfe5fbcf70eb.png)


![image](https://user-images.githubusercontent.com/113626253/236295598-c6170813-671a-43af-8f0f-8ac641469a2a.png)


![image](https://user-images.githubusercontent.com/113626253/236295649-9234f6b3-89ff-4bbc-87cb-85186fefc77a.png)


![image](https://user-images.githubusercontent.com/113626253/236295680-71d7f39e-6c19-4376-8a74-925bbc7496cc.png)


![image](https://user-images.githubusercontent.com/113626253/236295723-ff9fb563-bf52-4876-b46b-7be19e68a6a4.png)


![image](https://user-images.githubusercontent.com/113626253/236295755-cd87ec3e-adc1-4b12-9928-50f79a5d83ea.png)


![image](https://user-images.githubusercontent.com/113626253/236295802-3a02fa33-c595-401c-8d45-5d6a804884ed.png)


![image](https://user-images.githubusercontent.com/113626253/236295848-fae62325-7f1e-4503-9301-b86cd0d217f6.png)


![image](https://user-images.githubusercontent.com/113626253/236295893-320b6da2-d705-43a6-ba37-7009e62e8c2b.png)


![image](https://user-images.githubusercontent.com/113626253/236295930-7a4aceef-7622-4c36-a580-c40f00fc3f63.png)


![image](https://user-images.githubusercontent.com/113626253/236295981-4fe9d7b8-2720-4393-a367-65cfef47976f.png)


![image](https://user-images.githubusercontent.com/113626253/236296036-d056d26d-68dd-435f-aad5-70eb3f8bca7b.png)


![image](https://user-images.githubusercontent.com/113626253/236296068-846f6d1b-0030-4001-8ee1-52edd0fc752b.png)


![image](https://user-images.githubusercontent.com/113626253/236296096-8ac9e9cb-76be-4734-bb53-ced47431c4a0.png)


![image](https://user-images.githubusercontent.com/113626253/236296125-1b8d937f-cd45-4443-b81d-ad4b187395fe.png)


![image](https://user-images.githubusercontent.com/113626253/236296153-e994b6d5-3bb7-4519-acf9-2637c4a4e62d.png)


![image](https://user-images.githubusercontent.com/113626253/236296185-49d547c0-a290-4fe4-ac5a-dd33d7d69571.png)


![image](https://user-images.githubusercontent.com/113626253/236296209-c596f211-10e6-4f10-b0fe-f6ce0e94443e.png)


![image](https://user-images.githubusercontent.com/113626253/236296234-77a92ea2-35c5-4b2d-944c-f59f9beb6c8c.png)


![image](https://user-images.githubusercontent.com/113626253/236296267-db8c7e8a-6883-4d2b-8497-c905adf12192.png)


![image](https://user-images.githubusercontent.com/113626253/236296317-47fb97f3-7590-481a-b2c8-85f0b8d342e6.png)


![image](https://user-images.githubusercontent.com/113626253/236296378-fb999767-a149-4c57-b14b-b02ee19a64f3.png)


Without going into too much detail about these plots it was interesting to see the coordination between the continuous variables and the response based on different factors such as strength state and shot type. Shot distance tended to highly correlated with most variables including goals. Furthermore, there was a relatively large negative correlation between shot angle and shot distance. These plots gave a better understanding of the predictors and how they relate to each other before entering the modeling phase. One thing to note was no transformations were added to the predictors in this variation of the model. This will be experimented with in future work.

![image](https://user-images.githubusercontent.com/113626253/236296509-c3fe11fe-2452-44a4-912e-f3c56f7dc1da.png)


The plot above show more information about the correlation between the variables including the binary encoded categorical features. Similarly to the other plots, the shot distance is highly correlated to several other features. The larger the square the more they are correlated. The blue represents a positive correlation and the red represents a negative correlation. In addition to the plots a redundancy analysis was run over each predictor to check if we should remove any column before modeling. Based on the output takeaway last was redundant to the model but since there was nothing similar knowing the context I decided to keep it in the model. This could be experimented with in future work.

## fit model
XG_model <- xgb.train(data = dtrain, max.depth = 6, eta = 0.005, nrounds = 5000,
               nthread = 4, watchlist = watch_list, eval_metric = "auc",
               eval_metric = "logloss", early_stopping_rounds = 60,
               verbose = 0, objective = "binary:logistic")

## save model
xgb.save(XG_model, 'XG_model')

After splitting the data into the training, testing, and holdout data the model was fit using extreme gradient boosting. This method creates many decision trees and chains each of them together over hundreds of iterations. This helps correct for misclassification as more trees are built boosting the performance of the model. We can also tune the model based on the depth of the tree and the model learning rate. In general the smaller the eta (learning rate) the better the performance and longer the model takes to train. A deeper tree could cause over fitting meaning the model is very good at predicting on the training data but will not do as well when it sees new data. Ideally the model should be good at predicting on unseen data. We can test how the model does on new observations by using the test data. If the performance of the model is similar between the training and testing data then we do not have to worry about it being over fit. The model was run for 5,000 iterations with the model set to stop running when the log loss did not improve over a 60 iteration span. This helps prevent over fitting the model.

![image](https://user-images.githubusercontent.com/113626253/236296735-4eb50809-848a-4059-8908-04ec690a090c.png)


![image](https://user-images.githubusercontent.com/113626253/236296772-7b9782fb-938c-4265-8080-a79b95938dbc.png)


![image](https://user-images.githubusercontent.com/113626253/236296806-e2a2541e-3fa9-442a-9a80-fe3f616ff17a.png)


Looking at the area under the curve for both sets of data we can see that the model preformed relatively well. There was a slight difference in AUC as the iterations progressed but only by 0.01. The log loss was very close throughout the iterations. The feature importance was also plotted showing that the time between events and the shot distance were the most important in predicting expected goals.

![image](https://user-images.githubusercontent.com/113626253/236296871-8f342f0a-98f5-4de8-bc96-39597b65bd51.png)


![image](https://user-images.githubusercontent.com/113626253/236296907-5498068a-d011-416a-a712-5f03f6346a92.png)


![image](https://user-images.githubusercontent.com/113626253/236296952-f8ad0c23-d7ee-4e4a-8abf-7c0d6a0b27af.png)


![image](https://user-images.githubusercontent.com/113626253/236297005-6c64d7b1-229b-46f4-adcd-064ff6c580cd.png)


![image](https://user-images.githubusercontent.com/113626253/236297049-3cd7a7db-84d9-4eff-adb9-fb7eb3943287.png)


![image](https://user-images.githubusercontent.com/113626253/236297082-63bb7880-bf34-480b-bd3e-59a22da41fde.png)


![image](https://user-images.githubusercontent.com/113626253/236297111-56b835df-dca4-49c1-af96-55900b90d6ec.png)


![image](https://user-images.githubusercontent.com/113626253/236297136-caecd713-9511-450d-bdfc-c6a2258856a9.png)


![image](https://user-images.githubusercontent.com/113626253/236297169-98a33b66-4caa-4768-b6a1-44b1eb86f9a2.png)


![image](https://user-images.githubusercontent.com/113626253/236297210-48fb0234-ab4d-4f61-a2d1-57f26d1b9492.png)


The above plots show the relationship between the expected goals and various predictors. The black lines at the bottom of the plot show how many observations were in that range of the data. We would not want to draw conclusions on only a few data points. One example where this comes into play was the plot for the time difference. There were only a few times in the data that more than 150 seconds past between events thus it would be unwise to conclude anything beyond this range. The most telling plot in my opinion was of the shot distance. The probability of scoring decreases rapidly as the shot comes further from the net. Another interesting take away was the probability of scoring almost doubles on a rebound. It is important to note that these probabilities are extremely small so doubling a chance of scoring is still a low expected goal value by its self.

![image](https://user-images.githubusercontent.com/113626253/236297269-0d65a984-6260-4258-b70b-01efb55e0976.png)


The plot above shows how well the model preformed compared to the actual values. If the dotted line deviates from the grey area then the model either over predicted or under predicted the probabilities depending on the direction. In this case the model did a great job of predicting close to the actual values.

![image](https://user-images.githubusercontent.com/113626253/236297363-679cd61e-c534-43e6-8e15-f0d10f57a78e.png)


![image](https://user-images.githubusercontent.com/113626253/236297415-f952c2d3-1734-412c-9497-ce64556c8b49.png)


The model preformed pretty well on the new unseen data. It was expected to see a slight dip in performance. The area under the curve of 0.79 is still really solid and the log loss is very good. It is possible that the model is slightly over fit so that is something I will work on in my future version of my model.

![image](https://user-images.githubusercontent.com/113626253/236297531-47ab3f14-9114-49e8-bc50-b079e2348544.png)


![image](https://user-images.githubusercontent.com/113626253/236297564-e1a111a0-51ca-44dd-b3ab-e8f8695231f0.png)


![image](https://user-images.githubusercontent.com/113626253/236297608-94611fc5-1702-44b6-a717-2a44ed242fd3.png)


![image](https://user-images.githubusercontent.com/113626253/236297642-2d86e1b7-58c7-42ec-9678-01ab9b759dbc.png)


![image](https://user-images.githubusercontent.com/113626253/236297675-332395c6-08cc-44c9-abd8-cb7a1c66016b.png)


![image](https://user-images.githubusercontent.com/113626253/236297705-3972c16f-8411-4e3d-96c9-8ed0c71916b9.png)


![image](https://user-images.githubusercontent.com/113626253/236297729-c1a490b7-fc5d-4028-a45b-7ece2f3cc882.png)


![image](https://user-images.githubusercontent.com/113626253/236297764-cea9c1ea-bc50-4587-b082-84a8d51ce5d0.png)


![image](https://user-images.githubusercontent.com/113626253/236297797-4e2f92d6-a129-4c61-a467-c0fdce54b393.png)


One interesting thing to take away from these plots was as the time between events decreases the expected goals increase. This makes sense thinking about hockey. The less time the goalie has to react to a shot the more likely the player scores. On another note a shot on a high danger attempt has a much better chance of scoring then outside that area. Likewise, a rebound chance is significantly better than a non rebound chance. Lastly, something was defiantly different about the other strength category as it was vastly different than the known strengths. This will be examined closer in the near future.


Final thoughts:

In hockey there is so much uncertainty in each game. Players have hot streaks and scoring droughts. There are line up changes on a daily biases. There are too many factors to name that go into a player scoring. Some of these factors can not be measured such as the will to win a game or captain worthy leadership. But other aspects of the game can be captured in a unique way though data. Sometimes it confirms an already know fact and other times the data can raise new, complected questions. I feel that these types of models can help explain the game in a detailed way to provide even more information about what leads to scoring. Expected goals models could give context to players strengths or weakness in different aspects of the sport. Moreover, these types of models could show how a player is preforming on a more detailed level. Again, All models are wrong but some are useful. I hope that this model could be a useful one.







  
