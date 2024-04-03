###install.packages('dplyr')
###install.packages("TTR")
###install.packages("randomForest")
###install.packages("lubridate")
###install.packages("ggplot2")
###install.packages("keras")

library(dplyr)
library(TTR)
library(randomForest)
library(lubridate)
library(tree)
library(ggplot2)
library(keras)


# Read data from csv
data.KO = read.csv("KO.csv")
data.TSLA = read.csv("TSLA.csv")


names(data.KO)


add_lagged_features <- function(df) {
  # Adding lagged features for all predictors for last 5 trading days
  for (lag in 1:5) {
    df[paste0('close_lag', lag)] <- dplyr::lag(df$Adj.Close, lag)
    df[paste0('volume_lag', lag)] <- dplyr::lag(df$Volume, lag)
    df[paste0('high_lag', lag)] <- dplyr::lag(df$High, lag)
    df[paste0('low_lag', lag)] <- dplyr::lag(df$Low, lag)
    df[paste0('open_lag', lag)] <- dplyr::lag(df$Open, lag)
    df[paste0('macd_lag', lag)] <- dplyr::lag(df$macd, lag)
    df[paste0('signal_lag', lag)] <- dplyr::lag(df$signal, lag)
    df[paste0('hist_lag', lag)] <- dplyr::lag(df$histogram, lag)
  }
  return(df)
}

calculate_MACD <- function(data, n_fast = 12, n_slow = 26, n_signal = 9) {
  # Calculate MACD, a very popular technical indicator in stock market analysis
  fastEMA <- TTR::EMA(data$Adj.Close, n = n_fast)
  slowEMA <- TTR::EMA(data$Adj.Close, n = n_slow)
  macd <- fastEMA - slowEMA
  signal <- TTR::EMA(macd, n = n_signal)
  histogram <- macd - signal
  return(data.frame(macd, signal, histogram))
}


calculate_mape <- function(actuals, predictions) {
  # Ensure that actuals and predictions are of the same length
  if (length(actuals) != length(predictions)) {
    stop("Actuals and predictions must be of the same length")
  }
  
  # Calculate the absolute percentage errors
  ape <- abs((actuals - predictions) / actuals)
  
  # Remove infinite or missing values which can occur if actuals include zeros
  ape <- ape[is.finite(ape)]
  
  # Calculate and return the mean absolute percentage error
  mean(ape) * 100
}

macd_results = calculate_MACD(data.KO)
# If you encounter error at this step, ignore, it does affect results
data.KO$macd = macd_results$macd
data.KO$signal = macd_results$signal
data.KO$histogram = macd_results$histogram

data.KO = na.omit(data.KO)
data.KO.NN = data.KO
data.KO = add_lagged_features(data.KO)

data.KO$Date <- as.Date(data.KO$Date, format = "%Y-%m-%d")
data.KO$Year <- year(data.KO$Date)
data.KO$Month <- month(data.KO$Date)
data.KO$Day <- day(data.KO$Date)
# Treating date as features, given the data has included the pandemic and so many big events,
# I thought this might be a good idea. BTW it's much easier to do this in python

data.KO = na.omit(data.KO)
data.KO

names(data.KO)
# Should include  [1] "Date"        "Open"        "High"        "Low"         "Close"       "Adj.Close"   "Volume"      "macd"       
#[9] "signal"      "hist"        "close_lag1"  "volume_lag1" "high_lag1"   "low_lag1"    "open_lag1"   "macd_lag1"  
#[17] "signal_lag1" "hist_lag1"   "close_lag2"  "volume_lag2" "high_lag2"   "low_lag2"    "open_lag2"   "macd_lag2"  
#[25] "signal_lag2" "hist_lag2"   "close_lag3"  "volume_lag3" "high_lag3"   "low_lag3"    "open_lag3"   "macd_lag3"  
#[33] "signal_lag3" "hist_lag3"   "close_lag4"  "volume_lag4" "high_lag4"   "low_lag4"    "open_lag4"   "macd_lag4"  
#[41] "signal_lag4" "hist_lag4"   "close_lag5"  "volume_lag5" "high_lag5"   "low_lag5"    "open_lag5"   "macd_lag5"  
#[49] "signal_lag5" "hist_lag5"   "Year"        "Month"       "Day"     


data.KO.prep = subset(data.KO, select = -c(Date, Open, High, Low, Close, Volume, macd, signal, histogram))
# Making sure no future data is leaked
dim(data.KO.prep)
## SHould be 1221, 44

# First let's try a tree regression model
train <- data.KO.prep[1:976, ]
test <- data.KO.prep[(977:nrow(data.KO.prep)), ]
tree.KO = tree(Adj.Close~., data=train)

plot(tree.KO)
text(tree.KO, pretty=0)

tree.pred = predict(tree.KO, newdata = test)
mean((tree.pred - test$Adj.Close)^2)


tree.mape = calculate_mape(test$Adj.Close, tree.pred)
print(tree.mape)

#write.csv(tree.pred, "C:\\Users\\16784\\Documents\\R_Project\\Tree.csv", row.names=FALSE)

plot_data <- data.KO.prep[977:nrow(data.KO.prep), c("Day", "Month", "Year", "Adj.Close")]
plot_data$tree.pred <- tree.pred

# Visualizing trend
plot_data$Date <- with(plot_data, ISOdate(Year, Month, Day))
ggplot(plot_data, aes(x = Date)) +
  geom_line(aes(y = Adj.Close, colour = "Actual Price")) +
  geom_line(aes(y = tree.pred, colour = "Predicted Price")) +
  labs(title = "Actual vs Predicted Stock Prices Over Time",
       x = "Date", y = "Price") +
  scale_colour_manual("", 
                      breaks = c("Actual Price", "Predicted Price"),
                      values = c("blue", "red")) +
  theme_minimal()

# Visualizing residual
residuals <- data.KO$Adj.Close[test] - tree.pred.KO

plot(tree.pred.KO, residuals, xlab = "Predicted Values", ylab = "Residuals",
     main = "Residuals vs Predicted Values")
abline(h = 0, col = "red") 

hist(residuals, main = "Histogram of Residuals", xlab = "Residuals")



# Now let's try a more complicated one to see if it's anything better

split_ratio <- 0.8
split_index <- floor(nrow(data.KO.prep) * split_ratio)

train.data.KO <- data.KO.prep[1:split_index, ]
test.data.KO <- data.KO.prep[(split_index + 1):nrow(data.KO.prep), ]
# Train test split

target_var <- "Adj.Close"

# Prepare the data
train.labels <- train.data.KO[[target_var]]
train.data.KO <- train.data.KO[, !(names(train.data.KO) %in% target_var)]
test.KO <- test.data.KO[, !(names(test.data.KO) %in% target_var)]

# Train the model
rf.KO <- randomForest(x = train.data.KO, y = train.labels, importance = TRUE)

pred.KO <- predict(rf.KO, test.data.KO)

# Evaluation - you can use metrics like Mean Squared Error (MSE)
KO.price <- test.data.KO[[target_var]]

mse <- mean((pred.KO - KO.price)^2)
print(mse)
mape <- calculate_mape(KO.price, pred.KO)
print(mape)

test.data.KO$Date <- with(test.data.KO, ISOdate(Year, Month, Day))

#write.csv(pred.KO, "C:\\Users\\16784\\Documents\\R_Project\\RF.csv", row.names=FALSE)

test.data.KO$Predicted <- pred.KO

# Plotting Actual vs. Predicted Prices
library(ggplot2)
ggplot(test.data.KO, aes(x = Date)) +
  geom_line(aes(y = Adj.Close, colour = "Actual Price")) +
  geom_line(aes(y = Predicted, colour = "Predicted Price")) +
  labs(title = "Actual vs Predicted Stock Prices",
       x = "Date", y = "Stock Price") +
  scale_colour_manual("", 
                      breaks = c("Actual Price", "Predicted Price"),
                      values = c("blue", "red")) +
  theme_minimal()

# Residual plot
residuals <- test.data.KO$Adj.Close - test.data.KO$Predicted

ggplot(test.data.KO, aes(x = Predicted, y = residuals)) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Residuals vs Predicted", x = "Predicted Values", y = "Residuals") +
  theme_minimal()

# Feature importance plot
importance_values <- importance(rf.KO)
colnames(importance_values)
inc_mse_importance <- importance_values[, "%IncMSE"]
sorted_importance <- sort(inc_mse_importance, decreasing = TRUE)
top_n <- 15
top_sorted_importance <- head(sorted_importance, top_n)
barplot(top_sorted_importance, 
        main = "Feature Importance in KO RF Model",
        xlab = "% Increase in MSE",
        las = 2,  
        cex.names = 0.7,  
        horiz = TRUE)  

# RF demmontrated great imrpovement to simple tree regression
# And with the feature importance, especially the importance of Month, 
# we can assume some trend in the KO price is seasonal
# And that leads to the next choice of model: Recurrent Neural Network
# It allows largers moving window size, 5 is too small, and can capture sesonal trends


# Please note this requires a python environment, additional packages including
# numpy,tensorflow is required. Also the latest 3.12 version may not support tensorflow
# Consider create a virtual environment if needed

library(reticulate)
use_python("C:\\Users\\16784\\AppData\\Local\\Programs\\Python\\Python311", required = TRUE)

# Preprocessing: Regularization
# Speical handling: Since the year is just natural order sequence and yield a potential linear
# or logrithmatic relationship with price, we just treat it as other numeric values
# However, day and month are cyclical 

data.KO.NN
# We no longer needs lagged features.
data.KO.NN$Date <- as.Date(data.KO.NN$Date, format = "%Y-%m-%d")
data.KO.NN$Year <- year(data.KO.NN$Date)
data.KO.NN$Month <- month(data.KO.NN$Date)
data.KO.NN$Day <- day(data.KO.NN$Date)

data.KO.NN$day_sin <- sin((data.KO.NN$Day - 1) * 2 * pi / 31)
data.KO.NN$day_cos <- cos((data.KO.NN$Day - 1) * 2 * pi / 31)
data.KO.NN$month_sin <- sin((data.KO.NN$Month - 1) * 2 * pi / 12)
data.KO.NN$month_cos <- cos((data.KO.NN$Month - 1) * 2 * pi / 12)

data.KO.NN <- data.KO.NN %>%
  select(-Day, -Month, -Date)

numeric_vars <- sapply(data.KO.NN, is.numeric) & 
  !names(data.KO.NN) %in% c("day_sin", "day_cos", "month_sin", "month_cos")
data.KO.NN[numeric_vars] <- scale(data.KO.NN[numeric_vars])

data.KO.NN
dim(data.KO.NN)

set.seed(1213)  # for reproducibility

# Define the split ratio
split_ratio <- 0.8

# Calculate the index to split the data
split_index <- floor(nrow(data.KO.NN) * split_ratio)

# Split the data into training and testing sets
# Technically this part should be restructured as I changed my logic in the middle.
# So a portion of this is no longer needed.
# And there is a much easier way to do this
# But I'm too lazy to re-write this whole thing.

train <- data.KO.NN[1:split_index, ]
test <- data.KO.NN[(split_index + 1):nrow(data.KO.NN), ]

# Defining window size.
window_size <- 6
total_size <- nrow(data.KO.NN)

# Dimensions
num_train_sequences <- nrow(train) - window_size
num_features <- ncol(train) - 1  # Adjust based on actual number of features used

# Initialize arrays
train_X_array <- array(NA, dim = c(num_train_sequences, window_size, num_features))
train_Y_array <- array(NA, dim = num_train_sequences)

num_test_sequences <- nrow(test) - window_size

test_X_array <- array(NA, dim = c(num_test_sequences, window_size, num_features))
test_Y_array <- array(NA, dim = num_test_sequences)

num_train_sequences
num_test_sequences

# Populate training array
for (i in 1:num_train_sequences) {
  train_X_array[i, , ] <- as.matrix(train[i:(i + window_size - 1), -which(names(train) == 'Adj.Close')])
  train_Y_array[i] <- train[i + window_size, 'Adj.Close']
}

# Populate testing array
for (i in 1:num_test_sequences) {
  test_X_array[i, , ] <- as.matrix(test[i:(i + window_size - 1), -which(names(test) == 'Adj.Close')])
  test_Y_array[i] <- test[i + window_size, 'Adj.Close']
}

dim(train_X_array)

optimizer_adam(learning_rate = 0.000001)
RNN_KO <- keras_model_sequential() %>%
  layer_lstm(units = 144, return_sequences = TRUE, input_shape = c(6, 13)) %>%
  layer_dropout(rate = 0.7) %>%
  layer_lstm(units = 100, return_sequences = TRUE) %>%
  layer_dropout(rate = 0.6) %>%
  layer_lstm(units = 64, return_sequences = FALSE) %>%
  layer_dropout(rate = 0.6) %>%
  layer_dense(units = 36)
  layer_dense(units = 1, activation = 'linear')

RNN_KO %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam(),
)

history <- RNN_KO %>% fit(
  train_X_array, train_Y_array,
  epochs = 32,
  batch_size = 16,
  validation_split = 0.2
)

# Predictions and Evaluations
RNN_KO %>% evaluate(test_X_array, test_Y_array)

predictions <- RNN_KO %>% predict(test_X_array)

scaled_data <- scale(data.KO['Adj.Close'])
mean_adj_close <- attr(scaled_data, "scaled:center")
sd_adj_close <- attr(scaled_data, "scaled:scale")
original_scale_predictions <- predictions * sd_adj_close + mean_adj_close
original_scale_predictions <- original_scale_predictions[, 1]
original_scale_test_Y <- test_Y_array * sd_adj_close + mean_adj_close

mse <- mean((original_scale_predictions - original_scale_test_Y)^2)
print(mse)
mape <- calculate_mape(original_scale_test_Y, original_scale_predictions)
print(mape)
#write.csv(original_scale_predictions, "C:\\Users\\16784\\Documents\\R_Project\\RNN.csv", row.names=FALSE)





### Now the same but with TSLA data and tuned parameters


data.TSLA



macd_results = calculate_MACD(data.TSLA)
data.TSLA$macd = macd_results$macd
data.TSLA$signal = macd_results$signal
data.TSLA$histogram = macd_results$histogram

data.TSLA = na.omit(data.TSLA)
data.TSLA.NN = data.TSLA
data.TSLA = add_lagged_features(data.TSLA)

data.TSLA$Date <- as.Date(data.TSLA$Date, format = "%Y-%m-%d")
data.TSLA$Year <- year(data.TSLA$Date)
data.TSLA$Month <- month(data.TSLA$Date)
data.TSLA$Day <- day(data.TSLA$Date)

data.TSLA = na.omit(data.TSLA)

data.TSLA.prep = subset(data.TSLA, select = -c(Date, Open, High, Low, Close, Volume, macd, signal, histogram))
dim(data.TSLA.prep)

train <- data.TSLA.prep[1:976, ]
test <- data.TSLA.prep[(977:nrow(data.TSLA.prep)), ]
tree.TSLA = tree(Adj.Close~., data=train)

plot(tree.TSLA)
text(tree.TSLA, pretty=0)

tree.pred = predict(tree.TSLA, newdata = test)
mean((tree.pred - test$Adj.Close)^2)

tree.mape = calculate_mape(test$Adj.Close, tree.pred)
print(tree.mape)

#write.csv(tree.pred, "C:\\Users\\16784\\Documents\\R_Project\\Tree.csv", row.names=FALSE)

plot_data <- data.TSLA.prep[977:nrow(data.TSLA.prep), c("Day", "Month", "Year", "Adj.Close")]
plot_data$tree.pred <- tree.pred

# Visualizing trend
plot_data$Date <- with(plot_data, ISOdate(Year, Month, Day))
ggplot(plot_data, aes(x = Date)) +
  geom_line(aes(y = Adj.Close, colour = "Actual Price")) +
  geom_line(aes(y = tree.pred, colour = "Predicted Price")) +
  labs(title = "Actual vs Predicted Stock Prices Over Time",
       x = "Date", y = "Price") +
  scale_colour_manual("", 
                      breaks = c("Actual Price", "Predicted Price"),
                      values = c("blue", "red")) +
  theme_minimal()

# Visualizing residual
residuals <- data.TSLA$Adj.Close[test] - tree.pred.TSLA

plot(tree.pred.TSLA, residuals, xlab = "Predicted Values", ylab = "Residuals",
     main = "Residuals vs Predicted Values")
abline(h = 0, col = "red") 

hist(residuals, main = "Histogram of Residuals", xlab = "Residuals")



# Now let's try a more complicated one to see if it's anything better

split_ratio <- 0.8
split_index <- floor(nrow(data.TSLA.prep) * split_ratio)

train.data.TSLA <- data.TSLA.prep[1:split_index, ]
test.data.TSLA <- data.TSLA.prep[(split_index + 1):nrow(data.TSLA.prep), ]
# Train test split

target_var <- "Adj.Close"

# Prepare the data
train.labels <- train.data.TSLA[[target_var]]
train.data.TSLA <- train.data.TSLA[, !(names(train.data.TSLA) %in% target_var)]
test.TSLA <- test.data.TSLA[, !(names(test.data.TSLA) %in% target_var)]

# Train the model
rf.TSLA <- randomForest(x = train.data.TSLA, y = train.labels, importance = TRUE)

pred.TSLA <- predict(rf.TSLA, test.data.TSLA)

# Evaluation - you can use metrics like Mean Squared Error (MSE)
TSLA.price <- test.data.TSLA[[target_var]]

mse <- mean((pred.TSLA - TSLA.price)^2)
print(mse)
mape <- calculate_mape(TSLA.price, pred.TSLA)
print(mape)

test.data.TSLA$Date <- with(test.data.TSLA, ISOdate(Year, Month, Day))

#write.csv(pred.TSLA, "C:\\Users\\16784\\Documents\\R_Project\\RF.csv", row.names=FALSE)

test.data.TSLA$Predicted <- pred.TSLA

# Plotting Actual vs. Predicted Prices
library(ggplot2)
ggplot(test.data.TSLA, aes(x = Date)) +
  geom_line(aes(y = Adj.Close, colour = "Actual Price")) +
  geom_line(aes(y = Predicted, colour = "Predicted Price")) +
  labs(title = "Actual vs Predicted Stock Prices",
       x = "Date", y = "Stock Price") +
  scale_colour_manual("", 
                      breaks = c("Actual Price", "Predicted Price"),
                      values = c("blue", "red")) +
  theme_minimal()

# Residual plot
residuals <- test.data.TSLA$Adj.Close - test.data.TSLA$Predicted

ggplot(test.data.TSLA, aes(x = Predicted, y = residuals)) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Residuals vs Predicted", x = "Predicted Values", y = "Residuals") +
  theme_minimal()

# Feature importance plot
importance_values <- importance(rf.TSLA)
colnames(importance_values)
inc_mse_importance <- importance_values[, "%IncMSE"]
sorted_importance <- sort(inc_mse_importance, decreasing = TRUE)
top_n <- 15
top_sorted_importance <- head(sorted_importance, top_n)
barplot(top_sorted_importance, 
        main = "Feature Importance in TSLA RF Model",
        xlab = "% Increase in MSE",
        las = 2,  
        cex.names = 0.7,  
        horiz = TRUE)  


use_python("C:\\Users\\16784\\AppData\\Local\\Programs\\Python\\Python311", required = TRUE)


data.TSLA.NN
data.TSLA.NN$Date <- as.Date(data.TSLA.NN$Date, format = "%Y-%m-%d")
data.TSLA.NN$Year <- year(data.TSLA.NN$Date)
data.TSLA.NN$Month <- month(data.TSLA.NN$Date)
data.TSLA.NN$Day <- day(data.TSLA.NN$Date)

data.TSLA.NN$day_sin <- sin((data.TSLA.NN$Day - 1) * 2 * pi / 31)
data.TSLA.NN$day_cos <- cos((data.TSLA.NN$Day - 1) * 2 * pi / 31)
data.TSLA.NN$month_sin <- sin((data.TSLA.NN$Month - 1) * 2 * pi / 12)
data.TSLA.NN$month_cos <- cos((data.TSLA.NN$Month - 1) * 2 * pi / 12)

data.TSLA.NN <- data.TSLA.NN %>%
  select(-Day, -Month, -Date)

numeric_vars <- sapply(data.TSLA.NN, is.numeric) & 
  !names(data.TSLA.NN) %in% c("day_sin", "day_cos", "month_sin", "month_cos")
data.TSLA.NN[numeric_vars] <- scale(data.TSLA.NN[numeric_vars])

data.TSLA.NN
dim(data.TSLA.NN)

set.seed(1213) 

split_ratio <- 0.8
split_index <- floor(nrow(data.TSLA.NN) * split_ratio)
train <- data.TSLA.NN[1:split_index, ]
test <- data.TSLA.NN[(split_index + 1):nrow(data.TSLA.NN), ]

# Defining window size.
window_size <- 6
total_size <- nrow(data.TSLA.NN)

# Dimensions
num_train_sequences <- nrow(train) - window_size
num_features <- ncol(train) - 1

# Initialize arrays
train_X_array <- array(NA, dim = c(num_train_sequences, window_size, num_features))
train_Y_array <- array(NA, dim = num_train_sequences)

num_test_sequences <- nrow(test) - window_size

test_X_array <- array(NA, dim = c(num_test_sequences, window_size, num_features))
test_Y_array <- array(NA, dim = num_test_sequences)

num_train_sequences
num_test_sequences

# Populate training array
for (i in 1:num_train_sequences) {
  train_X_array[i, , ] <- as.matrix(train[i:(i + window_size - 1), -which(names(train) == 'Adj.Close')])
  train_Y_array[i] <- train[i + window_size, 'Adj.Close']
}

# Populate testing array
for (i in 1:num_test_sequences) {
  test_X_array[i, , ] <- as.matrix(test[i:(i + window_size - 1), -which(names(test) == 'Adj.Close')])
  test_Y_array[i] <- test[i + window_size, 'Adj.Close']
}

dim(train_X_array)

optimizer_adam(learning_rate = 0.000001)
RNN_TSLA <- keras_model_sequential() %>%
  layer_lstm(units = 144, return_sequences = TRUE, input_shape = c(6, 13)) %>%
  layer_dropout(rate = 0.7) %>%
  layer_lstm(units = 96, return_sequences = TRUE) %>%
  layer_dropout(rate = 0.6) %>%
  layer_lstm(units = 64, return_sequences = FALSE) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 32)
layer_dense(units = 1)

RNN_TSLA %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam(),
)

history <- RNN_TSLA %>% fit(
  train_X_array, train_Y_array,
  epochs = 8,
  batch_size = 16,
  validation_split = 0.2
)

# Predictions and Evaluations
RNN_TSLA %>% evaluate(test_X_array, test_Y_array)

predictions <- RNN_TSLA %>% predict(test_X_array)

scaled_data <- scale(data.TSLA['Adj.Close'])
mean_adj_close <- attr(scaled_data, "scaled:center")
sd_adj_close <- attr(scaled_data, "scaled:scale")
original_scale_predictions <- predictions * sd_adj_close + mean_adj_close
original_scale_predictions <- original_scale_predictions[, 1]
original_scale_test_Y <- test_Y_array * sd_adj_close + mean_adj_close

mse <- mean((original_scale_predictions - original_scale_test_Y)^2)
print(mse)
mape <- calculate_mape(original_scale_test_Y, original_scale_predictions)
print(mape)
