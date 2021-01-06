#Installation of required packages:

if(!require(dplyr)) install.packages("dplyr", 
                                     repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", 
                                         repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret",                                   #Additional packages will be required for 
                                     repos = "http://cran.us.r-project.org")    #running machine learning models. A prompt for
                                                                                #the installation of those will be given while
                                                                                #the caret "train" command is executed.
                                                                                #Entering "Yes" as the response will install
                                                                                #the required package.
if(!require(data.table)) install.packages("data.table",                          
                                        repos = "http://cran.us.r-project.org") 
if(!require(lubridate)) install.packages("lubridate", 
                                         repos = "http://cran.us.r-project.org")
if(!require(tidytext)) install.packages("tidytext", 
                                        repos = "http://cran.us.r-project.org")
if(!require(purrr)) install.packages("purrr", 
                                     repos = "http://cran.us.r-project.org")
if(!require(stringr)) install.packages("stringr", 
                                       repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra",                            
                                         repos = "http://cran.us.r-project.org"). 


#Loading required libraries for the script:

library(dplyr)
library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(tidytext)
library(purrr)
library(stringr)
library(gridExtra)

#Generating the "edx" and the "validation" dataset as per the provided script:

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", 
                             readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), 
                          "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, 
                                  list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

validation <- temp %>% semi_join(edx, by = "movieId") %>% semi_join(edx, 
                                                                  by = "userId")

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#Examining the structure of the "edx" dataset:
str(edx)
head(edx)

#Defining the RMSE(root mean square error) function:
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#For the data provided, considering the rating to be the variable for which 
#predictions are to be generated, the other variables can be considered as 
#potential sources of variation for determining the rating.

#Thus, the data can be examined for movie effect bias, user effect bias, 
#rating year effect bias, movie release year effect bias, and genre effect bias.

#A probable linear model for the data could be:
      # y = mu + b_i + b_u + b_ry + b_my + b_g + epsilon

#where, y = predicted rating,
#      mu = average rating,        
#     b_i = movie effect bias.
#     b_u = user effect bias,
#    b_ry = rating year effect bias,
#    b_my = movie release year effect bias,
#     b_g = genre bias.
# epsilon = independent errors from sampling from the same dataset

#Generating the "edx_dated" dataset which has the movie release year, and the 
#year in which the rating was provided, as additional variables:

edx_dated <- edx %>% mutate(rating_date = as_datetime(timestamp)) %>% 
  mutate(rating_year = as.numeric(year(rating_date)), 
         movie_year = as.numeric(str_sub(title, start = -5L, end = -2L)))

str(edx_dated)
head(edx_dated)

#Graph showing movie effect on ratings:
M <- edx_dated %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 100, color = "black", fill = "blue") + 
  scale_x_continuous(trans = "sqrt") +
  ggtitle("Movie Effect Variation")
M

#Graph showing user effect on ratings:
U <- edx_dated %>% 
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 100, color = "black", fill = "blue") + 
  scale_x_continuous(trans = "sqrt") +
  ggtitle("User Effect Variation")
U

#Graph showing rating year effect on ratings:
RY <- edx_dated %>% 
  count(rating_year) %>% 
  ggplot(aes(rating_year, n)) + 
  geom_col(color = "black", fill = "red") +
  scale_y_continuous(trans = "sqrt") +
  ggtitle("Rating Year Effect Variation")
RY

#Graph showing movie release year effect on ratings:
MY <- edx_dated %>% 
  count(movie_year) %>% 
  ggplot(aes(movie_year, n)) + 
  geom_col(color = "black", fill = "blue") + 
  scale_y_continuous(trans = "sqrt") +
  ggtitle("Release Year Effect Variation")
MY


#A movie may have one or more genres. For a movie belonging to multiple genres,
#all genres which define it are clubbed together in the "genres" column in the 
#"edx" dataset. To quantify the genre effect, with some data wrangling, unique
#genres present in the whole set can be determined, and a score can be assigned 
#to each. Then, a "genre_score" variable can be defined such that scores of all  
#genres under which the movie is categorized will be added up to give a
#total genre score by which the genre effect on the movie in concern is to be 
#assessed.

#Creating a dataset for determining the genre effect:
edx_genre <- edx %>% select(rating, title, genres)

#Dataframe containing unique genres:
genres <- edx_genre %>% group_by(genres) %>% summarise(n = n()) %>% 
  unnest_tokens(genre, genres) %>% pull(genre) %>% unique() %>% as.character()

genres[1] <- "(no genres listed)"                                               #Certain corrections are required to be done 
genres <- genres[-(2:3)]                                                        #manually to the character vector of unique  
genres[9] <- "Sci-Fi"                                                           #genres for special cases such as where the genre  
genres <- genres[-10]                                                           #name is hyphenated.
genres[20] <- "Film-Noir"
genres <- genres[-21]
genres[2:7] <- str_to_title(genres[2:7])
genres[8] <- str_to_upper(genres[8])
genres[10:19] <- str_to_title(genres[10:19])
genres

#Dataframe defining genre scores:
genre_score_chart <- data.frame(genres = genres, score = (1:20))                #Scores are determined arbitrarily simply based 
                                                                                #on the order of the dataframe. The total will
                                                                                #still serve as a nominal number which is 
                                                                                #representative of all genres that the movie
                                                                                #belongs to.

#Function to define genre scores.
genre_score_fun <- function(g){
  g_dat <- g %>% str_split(pattern = "\\|") %>% .[[1]] %>% data.frame(genre = .)
  genre_score <- sapply(g_dat, function(genre){
    ind <- match(genre, genre_score_chart$genres)
    genre_score_chart$score[ind]
  }
  )
  sum(genre_score)
}

#Dataframe with genre scores:
edx_genre_scored <- edx_genre %>% 
  mutate(genre_score = sapply(genres, genre_score_fun))

str(edx_genre_scored)
head(edx_genre_scored)

#Graph showing genre effect on ratings:
G <- edx_genre_scored %>% 
  count(genre_score) %>% 
  ggplot(aes(genre_score, n)) + 
  geom_col(color = "black", fill = "yellow") +
  scale_y_continuous(trans = "sqrt") +
  ggtitle("Variation Based on Genres")
G


#Graphs showing variations based on all examined variables:
variability_graphs <- grid.arrange(M, U, MY, RY, G, ncol = 2)
variability_graphs

#As the effect of the year in which the rating is provided does not seem to show 
#much variability, that variable can be excluded, and a linear model such as the
#following can be fit to the data:

# y = mu + b_i + b_u + b_y + b_g + epsilon

#where, y = predicted rating,
#      mu = average rating,        
#     b_i = movie effect bias.
#     b_u = user effect bias,
#     b_y = movie release year effect bias,
#     b_g = genre bias.
# epsilon = independent errors from sampling from the same dataset

#Creating a dataset with only the variables to be considered for rating 
#prediction:

edx_pred_dat <- edx %>% mutate(movie_year = edx_dated$movie_year, 
                               genre_score = edx_genre_scored$genre_score) %>%
  select(rating, movieId, userId, movie_year, genre_score) %>%
  mutate(userId = as.numeric(userId), 
         genre_score = as.numeric(genre_score))

str(edx_pred_dat)
head(edx_pred_dat)

#Defining "mu", the average rating of the dataset:
mu <- mean(edx_pred_dat$rating)

#Generating "pred_b_i", prediction based on movie effect, "b_i":
movie_avgs <- edx_pred_dat %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

edx_pred_dat <- edx_pred_dat %>% left_join(movie_avgs, by='movieId')
edx_pred_dat <- edx_pred_dat %>% mutate(pred_b_i = mu + b_i)

#Generating "pred_b_u", prediction based on user effect, "b_u", along with the
#previous bias effects:
user_avgs <- edx_pred_dat %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i.y))

edx_pred_dat <- edx_pred_dat %>% left_join(user_avgs, by='userId')
edx_pred_dat <- edx_pred_dat %>% mutate(pred_b_u = mu + b_i + b_u)

#Generating "pred_b_y", prediction based on movie release year effect, "b_y", 
#along with the previous bias effects:
year_avgs <- edx_pred_dat %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>% 
  group_by(movie_year) %>%
  summarize(b_y = mean(rating - mu - b_i.y - b_u.y))

edx_pred_dat <- edx_pred_dat %>% left_join(year_avgs, by='movie_year')
edx_pred_dat <- edx_pred_dat %>% mutate(pred_b_y = mu + b_i + b_u + b_y)

#Generating "pred_b_g", prediction based on genre effect, "b_g", along with the
#previous bias effects:
category_avgs <- edx_pred_dat %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(year_avgs, by='movie_year') %>%
  group_by(genre_score) %>%
  summarize(b_g = mean(rating - mu - b_i.y - b_u.y - b_y.y))

edx_pred_dat <- edx_pred_dat %>% left_join(category_avgs, by='genre_score')
edx_pred_dat <- edx_pred_dat %>% mutate(pred_b_g = mu + b_i + b_u + b_y + b_g)


#The "edx_pred_dat" dataset now contains all variables required for the linear
#model:
# y = mu + b_i + b_u + b_y + b_g + epsilon

str(edx_pred_dat)
head(edx_pred_dat)

#All these effects can be further subject to regularization using the 
#regularization parameter "lambda". 

#To obtain the optimal value for lambda, a representative sample of 10000
#observations can be used, considering limitations of computing capacity.

#The "edx" and the "validation" dataset each have the same 10 unique ratings:
edx_ratings <- edx %>% pull(rating) %>% unique()
validation_ratings <- edx %>% pull(rating) %>% unique()
identical(edx_ratings, validation_ratings)

#Thus, an appropriately representative sample can be constructed with a 1000 
#random samples of each rating as follows:

sample_5 <- edx_pred_dat %>% filter(rating == 5.0) %>% sample_n(1000)
sample_4_5 <- edx_pred_dat %>% filter(rating == 4.5) %>% sample_n(1000)
sample_4 <- edx_pred_dat %>% filter(rating == 4.0) %>% sample_n(1000)
sample_3_5 <- edx_pred_dat %>% filter(rating == 3.5) %>% sample_n(1000)
sample_3 <- edx_pred_dat %>% filter(rating == 3.0) %>% sample_n(1000)
sample_2_5 <- edx_pred_dat %>% filter(rating == 2.5) %>% sample_n(1000)
sample_2 <- edx_pred_dat %>% filter(rating == 2.0) %>% sample_n(1000)
sample_1_5 <- edx_pred_dat %>% filter(rating == 1.5) %>% sample_n(1000)
sample_1 <- edx_pred_dat %>% filter(rating == 1.0) %>% sample_n(1000)
sample_0_5 <- edx_pred_dat %>% filter(rating == 0.5) %>% sample_n(1000)


#Representative sample set:
sample_set <- rbind(sample_5, sample_4_5, sample_4, sample_3_5, sample_3,
                    sample_2_5, sample_2, sample_1_5, sample_1, sample_0_5) 

#Sample set rating column:
y = sample_set$rating

#Creating training and test sets for lambda optimization:
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y, times = 1, p = 0.1, list = FALSE)
train_set <- sample_set[-test_index,]
temp <- sample_set[test_index,]

test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)

#Defining range of lambdas:
lambdas <- seq(0, 10, 0.25)

#Function calculating root mean square errors over the range of lambdas to be
#tested:
rmses <- sapply(lambdas, function(l){
  
  b_i <- train_set %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  
  b_u <- train_set %>% 
    left_join(b_i, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i.y)/(n()+l))
  
  b_y <- train_set %>% 
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(movie_year) %>%
    summarize(b_y = sum(rating - mu - b_i.y - b_u.y)/(n()+l))
  
  b_g <- train_set %>% 
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_y, by='movie_year') %>%
    group_by(genre_score) %>%
    summarize(b_y = sum(rating - mu - b_i.y - b_u.y - b_y.y)/(n()+l))
  
  predicted_ratings <- test_set %>% 
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    left_join(b_y, by='movie_year') %>%
    left_join(b_g, by='genre_score') %>%
    mutate(pred = mu + b_i.y + b_u.y + b_y.y + b_g) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

#Graph showing values of root mean square errors plotted against different 
#values of lambda.
lambda_plot <- qplot(lambdas, rmses)
lambda_plot

#Optimal lambda value will yield minimum RMSE:
lambda <- lambdas[which.min(rmses)]

#Optimal lambda value:
lambda

#Lambda Optimization Graph:
plot(lambdas, rmses, type = "b", xlab = "Lambda Value") 
title(main = "Lambda Optimization Curve")
points(lambdas, rmses, cex = .5, col = "dark red")
points(lambda , min(rmses), cex = 3, col = "red")
lines(lambdas, rmses, col = "blue")
text(x = lambda, y = min(rmses) + 0.04, labels = "3.25") 
text(x = lambda, y = min(rmses) + 0.02, labels = "Optimal Lambda Value")


#Generating a dataset with the variables considered for prediction with 
#regularized values using the regularization parameter lambda:

edx_pred_dat_l <- edx_pred_dat

movie_avgs_l <- edx_pred_dat_l %>% 
  group_by(movieId) %>% 
  summarize(b_i_l = mean(rating - mu)/(n()+lambda))

edx_pred_dat_l <- edx_pred_dat_l %>% left_join(movie_avgs_l, by='movieId')
edx_pred_dat_l <- edx_pred_dat_l %>% mutate(pred_b_i_l = mu + b_i_l) 

user_avgs_l <- edx_pred_dat_l %>% 
  left_join(movie_avgs_l, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u_l = mean(rating - mu - b_i_l.y)/(n()+lambda))

edx_pred_dat_l <- edx_pred_dat_l %>% left_join(user_avgs_l, by='userId')
edx_pred_dat_l <- edx_pred_dat_l %>% mutate(pred_b_u_l = mu + b_i_l + b_u_l)

year_avgs_l <- edx_pred_dat_l %>% 
  left_join(movie_avgs_l, by='movieId') %>%
  left_join(user_avgs_l, by='userId') %>% 
  group_by(movie_year) %>%
  summarize(b_y_l = mean(rating - mu - b_i_l.y - b_u_l.y)/(n()+lambda))

edx_pred_dat_l <- edx_pred_dat_l %>% left_join(year_avgs_l, by='movie_year')
edx_pred_dat_l <- edx_pred_dat_l %>% mutate(pred_b_y_l = mu + b_i_l + b_u_l + 
                                              b_y_l)

edx_pred_dat_l_j1 <- edx_pred_dat_l %>% 
  left_join(movie_avgs_l, by='movieId')

edx_pred_dat_l_j2 <- edx_pred_dat_l_j1 %>%
  left_join(user_avgs_l, by='userId') 

edx_pred_dat_l_j3 <- edx_pred_dat_l_j2 %>%  
  left_join(year_avgs_l, by='movie_year')

category_avgs_l <-  edx_pred_dat_l_j3 %>% group_by(genre_score) %>%
  summarize(b_g_l = mean(rating - mu - b_i_l.y - b_u_l.y - b_y_l.y)/(n() + 
                                                                       lambda))

edx_pred_dat_l <- edx_pred_dat_l %>% left_join(category_avgs_l, 
                                               by='genre_score')
edx_pred_dat_l <- edx_pred_dat_l %>% mutate(pred_b_g_l = mu + b_i_l + b_u_l + 
                                              b_y_l + b_g_l)


#Dataset with the variables considered for prediction with regularized  
#values using the regularization parameter lambda:

str(edx_pred_dat_l)
head(edx_pred_dat_l)


#Using a random sampling approach, considering limitations of computing time and 
#computing capacity, 1000 randomly chosen samples are taken from the "edx" 
#dataset for training and testing the machine learning algorithms. The 
#predictions obtained for the average rating of each movie from these algorithms 
#will be checked against the average ratings for the same movies in the 
#validation dataset in order to obtain the final RMSE value.

#There will be a total of six replications of the sampling procedure with the
#seed set at different points, and independent training and testing for each
#sample, in order to demonstrate consistent reproducibility of results obtained 
#by this method.

#The final result will include the minimum RMSE obtained from all prediction 
#data, and the average RMSE of all models being run on all samples.



#Sample 1:
#Creating training and test sets for machine learning algorithm training:
set.seed(1, sample.kind="Rounding")
ml_sample_1 <- edx_pred_dat_l %>% sample_n(1000)

ml_samp_movieids <- ml_sample_1$movieId %>% unique()

ml_rating <- edx_pred_dat_l %>% filter(movieId %in% ml_samp_movieids) %>% 
  group_by(movieId) %>% 
  summarise(avg_rating = mean(rating))

ml_rating_df <- as.data.frame(ml_rating)

ml_sample_set <- inner_join(ml_rating_df, ml_sample_1, by = "movieId")

y <- ml_sample_set$rating

set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)
ml_train_set <- ml_sample_set[-test_index,]
ml_test_set <- ml_sample_set[test_index,]

rm(test_index)

#Machine Learning training data using various bias effects and predictions 
#based on them as predictors:

ml_dat <- ml_train_set %>% mutate(y = avg_rating) %>% select(-c(rating, 
                                                                avg_rating,
                                                                movieId, userId, 
                                                                movie_year,
                                                                genre_score))

#Machine Learning test data using various bias effects and predictions 
#based on them as predictors:

ml_dat_test <- ml_test_set %>% mutate(y = avg_rating) %>% select(-c(rating, 
                                                                avg_rating,
                                                                movieId, userId, 
                                                                movie_year,
                                                                genre_score))


#Principal Component Analysis of machine learning training data:

pca_1 <- prcomp(ml_dat)

summary(pca_1)
plot(pca_1$sdev, type = "b", xlab = "Principal Component Number")
title(main = "Principal Component Analysis #1")
points(pca_1$sdev, cex = .5, col = "dark red")
points(6 , pca_1$sdev[6], cex = 3, col = "red")
lines(pca_1$sdev, col = "blue")
axis(1, 0:17, col.axis = "blue")
text(x = 6, y = 0.2, labels = "PC6") 
text(x = 6, y = 0.1, labels = "Full Cumulative Proportion Of Variance")

#The first six principal components explain all the variability of the data
#as seen from the summary and the graph:

#Ratings for the algorithms to be trained on:
y <- ml_train_set$avg_rating

#Creating a dataframe with the first six principal components as predictors:
pca_dat <- data.frame(y = ml_train_set$avg_rating, x = pca_1$x[,1:6])


#Using the "caret" package training models to generate predictions and test
#the resulting RMSEs with testing data:

set.seed(1, sample.kind="Rounding")
bayesglm_fit <- train(y ~ ., method = "bayesglm", data = pca_dat)
pred_bayesglm <- predict(bayesglm_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_bayesglm)

set.seed(1, sample.kind="Rounding")
gaussprLinear_fit <- train(y ~ ., method = "gaussprLinear", data = pca_dat)
pred_gaussprLinear <- predict(gaussprLinear_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_gaussprLinear)

set.seed(1, sample.kind="Rounding")
glm_fit <- train(y ~ ., method = "glm", data = pca_dat)
pred_glm <- predict(glm_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_glm)

set.seed(1, sample.kind="Rounding")
glmStepAIC_fit <- train(y ~ ., method = "glmStepAIC", data = pca_dat)
pred_glmStepAIC <- predict(glmStepAIC_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_glmStepAIC)

set.seed(1, sample.kind="Rounding")
glmnet_fit <- train(y ~ ., method = "glmnet", data = pca_dat)
pred_glmnet <- predict(glmnet_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_glmnet)

set.seed(1, sample.kind="Rounding")
pls_fit <- train(y ~ ., method = "pls", data = pca_dat)
pred_pls <- predict(pls_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_pls)

set.seed(1, sample.kind="Rounding")
simpls_fit <- train(y ~ ., method = "simpls", data = pca_dat)
pred_simpls <- predict(simpls_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_simpls)

set.seed(1, sample.kind="Rounding")
kernelpls_fit <- train(y ~ ., method = "kernelpls", data = pca_dat)
pred_kernelpls <- predict(kernelpls_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_kernelpls)

set.seed(1, sample.kind="Rounding")
widekernelpls_fit <- train(y ~ ., method = "widekernelpls", data = pca_dat)
pred_widekernelpls <- predict(widekernelpls_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_widekernelpls)

#Using an ensemble model to to generate predictions and test
#the resulting RMSEs with testing data:

set.seed(1, sample.kind="Rounding")
randomGLM_fit <- train(y ~ ., method = "randomGLM", data = pca_dat)
pred_randomGLM <- predict(randomGLM_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_randomGLM)


#Creating a dataset with average ratings of the same movies as in the sample set,
#taken from the the validation set.

val_samp <- validation %>% filter(movieId %in% ml_test_set$movieId) %>% 
  group_by(movieId) %>% 
  summarise(avg_rating = mean(rating)) %>% pull(avg_rating)

#Calculating the RMSE for each individual model's prediction of average rating  
#for each movie, taken with the actual average rating of that movie in the 
#validation set.

RMSE(val_samp, pred_bayesglm)
RMSE(val_samp, pred_gaussprLinear)
RMSE(val_samp, pred_glm)
RMSE(val_samp, pred_glmStepAIC)
RMSE(val_samp, pred_glmnet)
RMSE(val_samp, pred_pls)
RMSE(val_samp, pred_simpls)
RMSE(val_samp, pred_kernelpls)
RMSE(val_samp, pred_widekernelpls)
RMSE(val_samp, pred_randomGLM)

#Dataframe with all models along with their corresponding RMSE:

final_rmses <- data.frame(Models = c("bayesglm", "gaussprLinear", "glm",
                                     "glmStepAIC", "glmnet", "pls",
                                     "simpls", "kernelpls", "widekernelpls", 
                                     "randomGLM"),
                          ML_RMSE = c(RMSE(val_samp, pred_bayesglm),
                                      RMSE(val_samp, pred_gaussprLinear),
                                      RMSE(val_samp, pred_glm),
                                      RMSE(val_samp, pred_glmStepAIC),
                                      RMSE(val_samp, pred_glmnet),
                                      RMSE(val_samp, pred_pls),
                                      RMSE(val_samp, pred_simpls),
                                      RMSE(val_samp, pred_kernelpls),
                                      RMSE(val_samp, pred_widekernelpls),
                                      RMSE(val_samp, pred_randomGLM)
                          ))

final_rmses_1 <- final_rmses %>% mutate(rmse_rounded = round(ML_RMSE, digits = 3))


#Graph showing RMSEs obtained by each model:

fr_1 <- final_rmses_1 %>% ggplot(aes(Models, ML_RMSE, fill = Models)) + geom_col() +
  geom_hline(yintercept = 0.67, lty = 2, size = 1) +
  geom_text(aes(x = 0, y = 0.67, label = "Average RMSE = 0.67"), nudge_x = 1.4,
            nudge_y = 0.02) +
  geom_text(aes(label = rmse_rounded), nudge_y = 0.05) +
  geom_text(aes(label = signif(ML_RMSE, digits = 6)), 
            angle = 90, nudge_y = -0.4) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
fr_1

#Minimum RMSE obtained:
min(final_rmses_1$ML_RMSE)

#Model yielding the minimal RMSE:
min_ind <- which.min(final_rmses_1$ML_RMSE)
final_rmses_1[min_ind,]

#Average RMSE from all models:
result_1 <- mean(final_rmses_1$ML_RMSE)
result_1


#Sample 2:
#Creating training and test sets for machine learning algorithm training:
set.seed(1000, sample.kind="Rounding")
ml_sample_2 <- edx_pred_dat_l %>% sample_n(1000)

ml_samp_movieids <- ml_sample_2$movieId %>% unique()

ml_rating <- edx_pred_dat_l %>% filter(movieId %in% ml_samp_movieids) %>% 
  group_by(movieId) %>% 
  summarise(avg_rating = mean(rating))

ml_rating_df <- as.data.frame(ml_rating)

ml_sample_set <- inner_join(ml_rating_df, ml_sample_2, by = "movieId")

y <- ml_sample_set$rating

set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)
ml_train_set <- ml_sample_set[-test_index,]
ml_test_set <- ml_sample_set[test_index,]

rm(test_index)

#Machine Learning training data using various bias effects and predictions 
#based on them as predictors:

ml_dat <- ml_train_set %>% mutate(y = avg_rating) %>% select(-c(rating, 
                                                                avg_rating,
                                                                movieId, userId, 
                                                                movie_year,
                                                                genre_score))

#Machine Learning test data using various bias effects and predictions 
#based on them as predictors:

ml_dat_test <- ml_test_set %>% mutate(y = avg_rating) %>% select(-c(rating, 
                                                                    avg_rating,
                                                                    movieId, userId, 
                                                                    movie_year,
                                                                    genre_score))


#Principal Component Analysis of machine learning training data:

pca_2 <- prcomp(ml_dat)

summary(pca_2)
plot(pca_2$sdev, type = "b", xlab = "Principal Component Number")
title(main = "Principal Component Analysis #2")
points(pca_2$sdev, cex = .5, col = "dark red")
points(6 , pca_2$sdev[6], cex = 3, col = "red")
lines(pca_2$sdev, col = "blue")
axis(1, 0:17, col.axis = "blue")
text(x = 6, y = 0.2, labels = "PC6") 
text(x = 6, y = 0.1, labels = "Full Cumulative Proportion Of Variance")

#The first six principal components explain all the variability of the data
#as seen from the summary and the graph:

#Ratings for the algorithms to be trained on:
y <- ml_train_set$avg_rating

#Creating a dataframe with the first six principal components as predictors:
pca_dat <- data.frame(y = ml_train_set$avg_rating, x = pca_2$x[,1:6])


#Using the "caret" package training models to generate predictions and test
#the resulting RMSEs with testing data:

set.seed(1, sample.kind="Rounding")
bayesglm_fit <- train(y ~ ., method = "bayesglm", data = pca_dat)
pred_bayesglm <- predict(bayesglm_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_bayesglm)

set.seed(1, sample.kind="Rounding")
gaussprLinear_fit <- train(y ~ ., method = "gaussprLinear", data = pca_dat)
pred_gaussprLinear <- predict(gaussprLinear_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_gaussprLinear)

set.seed(1, sample.kind="Rounding")
glm_fit <- train(y ~ ., method = "glm", data = pca_dat)
pred_glm <- predict(glm_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_glm)

set.seed(1, sample.kind="Rounding")
glmStepAIC_fit <- train(y ~ ., method = "glmStepAIC", data = pca_dat)
pred_glmStepAIC <- predict(glmStepAIC_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_glmStepAIC)

set.seed(1, sample.kind="Rounding")
glmnet_fit <- train(y ~ ., method = "glmnet", data = pca_dat)
pred_glmnet <- predict(glmnet_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_glmnet)

set.seed(1, sample.kind="Rounding")
pls_fit <- train(y ~ ., method = "pls", data = pca_dat)
pred_pls <- predict(pls_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_pls)

set.seed(1, sample.kind="Rounding")
simpls_fit <- train(y ~ ., method = "simpls", data = pca_dat)
pred_simpls <- predict(simpls_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_simpls)

set.seed(1, sample.kind="Rounding")
kernelpls_fit <- train(y ~ ., method = "kernelpls", data = pca_dat)
pred_kernelpls <- predict(kernelpls_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_kernelpls)

set.seed(1, sample.kind="Rounding")
widekernelpls_fit <- train(y ~ ., method = "widekernelpls", data = pca_dat)
pred_widekernelpls <- predict(widekernelpls_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_widekernelpls)

#Using an ensemble model to to generate predictions and test
#the resulting RMSEs with testing data:

set.seed(1, sample.kind="Rounding")
randomGLM_fit <- train(y ~ ., method = "randomGLM", data = pca_dat)
pred_randomGLM <- predict(randomGLM_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_randomGLM)


#Creating a dataset with average ratings of the same movies as in the sample set,
#taken from the the validation set.

val_samp <- validation %>% filter(movieId %in% ml_test_set$movieId) %>% 
  group_by(movieId) %>% 
  summarise(avg_rating = mean(rating)) %>% pull(avg_rating)

#Calculating the RMSE for each individual model's prediction of average rating  
#for each movie, taken with the actual average rating of that movie in the 
#validation set.

RMSE(val_samp, pred_bayesglm)
RMSE(val_samp, pred_gaussprLinear)
RMSE(val_samp, pred_glm)
RMSE(val_samp, pred_glmStepAIC)
RMSE(val_samp, pred_glmnet)
RMSE(val_samp, pred_pls)
RMSE(val_samp, pred_simpls)
RMSE(val_samp, pred_kernelpls)
RMSE(val_samp, pred_widekernelpls)
RMSE(val_samp, pred_randomGLM)

#Dataframe with all models along with their corresponding RMSE:

final_rmses <- data.frame(Models = c("bayesglm", "gaussprLinear", "glm",
                                     "glmStepAIC", "glmnet", "pls",
                                     "simpls", "kernelpls", "widekernelpls", 
                                     "randomGLM"),
                          ML_RMSE = c(RMSE(val_samp, pred_bayesglm),
                                      RMSE(val_samp, pred_gaussprLinear),
                                      RMSE(val_samp, pred_glm),
                                      RMSE(val_samp, pred_glmStepAIC),
                                      RMSE(val_samp, pred_glmnet),
                                      RMSE(val_samp, pred_pls),
                                      RMSE(val_samp, pred_simpls),
                                      RMSE(val_samp, pred_kernelpls),
                                      RMSE(val_samp, pred_widekernelpls),
                                      RMSE(val_samp, pred_randomGLM)
                          ))

final_rmses_2 <- final_rmses %>% mutate(rmse_rounded = round(ML_RMSE, digits = 3))


#Graph showing RMSEs obtained by each model:

fr_2 <- final_rmses_2 %>% ggplot(aes(Models, ML_RMSE, fill = Models)) + geom_col() +
  geom_hline(yintercept = 0.73, lty = 2, size = 1) +
  geom_text(aes(x = 0, y = 0.73, label = "Average RMSE = 0.73"), nudge_x = 1.4,
            nudge_y = 0.02) +
  geom_text(aes(label = rmse_rounded), nudge_y = 0.05) +
  geom_text(aes(label = signif(ML_RMSE, digits = 6)), 
            angle = 90, nudge_y = -0.4) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
fr_2

#Minimum RMSE obtained:
min(final_rmses_2$ML_RMSE)

#Model yielding the minimal RMSE:
min_ind <- which.min(final_rmses_2$ML_RMSE)
final_rmses_2[min_ind,]

#Average RMSE from all models:
result_2 <- mean(final_rmses_2$ML_RMSE)
result_2


#Sample 3:
#Creating training and test sets for machine learning algorithm training:
set.seed(5000, sample.kind="Rounding")
ml_sample_3 <- edx_pred_dat_l %>% sample_n(1000)

ml_samp_movieids <- ml_sample_3$movieId %>% unique()

ml_rating <- edx_pred_dat_l %>% filter(movieId %in% ml_samp_movieids) %>% 
  group_by(movieId) %>% 
  summarise(avg_rating = mean(rating))

ml_rating_df <- as.data.frame(ml_rating)

ml_sample_set <- inner_join(ml_rating_df, ml_sample_3, by = "movieId")

y <- ml_sample_set$rating

set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)
ml_train_set <- ml_sample_set[-test_index,]
ml_test_set <- ml_sample_set[test_index,]

rm(test_index)

#Machine Learning training data using various bias effects and predictions 
#based on them as predictors:

ml_dat <- ml_train_set %>% mutate(y = avg_rating) %>% select(-c(rating, 
                                                                avg_rating,
                                                                movieId, userId, 
                                                                movie_year,
                                                                genre_score))

#Machine Learning test data using various bias effects and predictions 
#based on them as predictors:

ml_dat_test <- ml_test_set %>% mutate(y = avg_rating) %>% select(-c(rating, 
                                                                    avg_rating,
                                                                    movieId, userId, 
                                                                    movie_year,
                                                                    genre_score))


#Principal Component Analysis of machine learning training data:

pca_3 <- prcomp(ml_dat)

summary(pca_3)
plot(pca_3$sdev, type = "b", xlab = "Principal Component Number")
title(main = "Principal Component Analysis #3")
points(pca_3$sdev, cex = .5, col = "dark red")
points(6 , pca_3$sdev[6], cex = 3, col = "red")
lines(pca_3$sdev, col = "blue")
axis(1, 0:17, col.axis = "blue")
text(x = 6, y = 0.2, labels = "PC6") 
text(x = 6, y = 0.1, labels = "Full Cumulative Proportion Of Variance")

#The first six principal components explain all the variability of the data
#as seen from the summary and the graph:

#Ratings for the algorithms to be trained on:
y <- ml_train_set$avg_rating

#Creating a dataframe with the first six principal components as predictors:
pca_dat <- data.frame(y = ml_train_set$avg_rating, x = pca_3$x[,1:6])


#Using the "caret" package training models to generate predictions and test
#the resulting RMSEs with testing data:

set.seed(1, sample.kind="Rounding")
bayesglm_fit <- train(y ~ ., method = "bayesglm", data = pca_dat)
pred_bayesglm <- predict(bayesglm_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_bayesglm)

set.seed(1, sample.kind="Rounding")
gaussprLinear_fit <- train(y ~ ., method = "gaussprLinear", data = pca_dat)
pred_gaussprLinear <- predict(gaussprLinear_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_gaussprLinear)

set.seed(1, sample.kind="Rounding")
glm_fit <- train(y ~ ., method = "glm", data = pca_dat)
pred_glm <- predict(glm_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_glm)

set.seed(1, sample.kind="Rounding")
glmStepAIC_fit <- train(y ~ ., method = "glmStepAIC", data = pca_dat)
pred_glmStepAIC <- predict(glmStepAIC_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_glmStepAIC)

set.seed(1, sample.kind="Rounding")
glmnet_fit <- train(y ~ ., method = "glmnet", data = pca_dat)
pred_glmnet <- predict(glmnet_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_glmnet)

set.seed(1, sample.kind="Rounding")
pls_fit <- train(y ~ ., method = "pls", data = pca_dat)
pred_pls <- predict(pls_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_pls)

set.seed(1, sample.kind="Rounding")
simpls_fit <- train(y ~ ., method = "simpls", data = pca_dat)
pred_simpls <- predict(simpls_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_simpls)

set.seed(1, sample.kind="Rounding")
kernelpls_fit <- train(y ~ ., method = "kernelpls", data = pca_dat)
pred_kernelpls <- predict(kernelpls_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_kernelpls)

set.seed(1, sample.kind="Rounding")
widekernelpls_fit <- train(y ~ ., method = "widekernelpls", data = pca_dat)
pred_widekernelpls <- predict(widekernelpls_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_widekernelpls)

#Using an ensemble model to to generate predictions and test
#the resulting RMSEs with testing data:

set.seed(1, sample.kind="Rounding")
randomGLM_fit <- train(y ~ ., method = "randomGLM", data = pca_dat)
pred_randomGLM <- predict(randomGLM_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_randomGLM)


#Creating a dataset with average ratings of the same movies as in the sample set,
#taken from the the validation set.

val_samp <- validation %>% filter(movieId %in% ml_test_set$movieId) %>% 
  group_by(movieId) %>% 
  summarise(avg_rating = mean(rating)) %>% pull(avg_rating)

#Calculating the RMSE for each individual model's prediction of average rating  
#for each movie, taken with the actual average rating of that movie in the 
#validation set.

RMSE(val_samp, pred_bayesglm)
RMSE(val_samp, pred_gaussprLinear)
RMSE(val_samp, pred_glm)
RMSE(val_samp, pred_glmStepAIC)
RMSE(val_samp, pred_glmnet)
RMSE(val_samp, pred_pls)
RMSE(val_samp, pred_simpls)
RMSE(val_samp, pred_kernelpls)
RMSE(val_samp, pred_widekernelpls)
RMSE(val_samp, pred_randomGLM)

#Dataframe with all models along with their corresponding RMSE:

final_rmses <- data.frame(Models = c("bayesglm", "gaussprLinear", "glm",
                                     "glmStepAIC", "glmnet", "pls",
                                     "simpls", "kernelpls", "widekernelpls", 
                                     "randomGLM"),
                          ML_RMSE = c(RMSE(val_samp, pred_bayesglm),
                                      RMSE(val_samp, pred_gaussprLinear),
                                      RMSE(val_samp, pred_glm),
                                      RMSE(val_samp, pred_glmStepAIC),
                                      RMSE(val_samp, pred_glmnet),
                                      RMSE(val_samp, pred_pls),
                                      RMSE(val_samp, pred_simpls),
                                      RMSE(val_samp, pred_kernelpls),
                                      RMSE(val_samp, pred_widekernelpls),
                                      RMSE(val_samp, pred_randomGLM)
                          ))

final_rmses_3 <- final_rmses %>% mutate(rmse_rounded = round(ML_RMSE, digits = 3))


#Graph showing RMSEs obtained by each model:

fr_3 <- final_rmses_3 %>% ggplot(aes(Models, ML_RMSE, fill = Models)) + geom_col() +
  geom_hline(yintercept = 0.69, lty = 2, size = 1) +
  geom_text(aes(x = 0, y = 0.69, label = "Average RMSE = 0.69"), nudge_x = 1.4,
            nudge_y = 0.02) +
  geom_text(aes(label = rmse_rounded), nudge_y = 0.05) +
  geom_text(aes(label = signif(ML_RMSE, digits = 6)), 
            angle = 90, nudge_y = -0.4) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
fr_3

#Minimum RMSE obtained:
min(final_rmses_3$ML_RMSE)

#Model yielding the minimal RMSE:
min_ind <- which.min(final_rmses_3$ML_RMSE)
final_rmses_3[min_ind,]

#Average RMSE from all models:
result_3 <- mean(final_rmses_3$ML_RMSE)
result_3


#Sample 4:
#Creating training and test sets for machine learning algorithm training:
set.seed(7000, sample.kind="Rounding")
ml_sample_4 <- edx_pred_dat_l %>% sample_n(1000)

ml_samp_movieids <- ml_sample_4$movieId %>% unique()

ml_rating <- edx_pred_dat_l %>% filter(movieId %in% ml_samp_movieids) %>% 
  group_by(movieId) %>% 
  summarise(avg_rating = mean(rating))

ml_rating_df <- as.data.frame(ml_rating)

ml_sample_set <- inner_join(ml_rating_df, ml_sample_4, by = "movieId")

y <- ml_sample_set$rating

set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)
ml_train_set <- ml_sample_set[-test_index,]
ml_test_set <- ml_sample_set[test_index,]

rm(test_index)

#Machine Learning training data using various bias effects and predictions 
#based on them as predictors:

ml_dat <- ml_train_set %>% mutate(y = avg_rating) %>% select(-c(rating, 
                                                                avg_rating,
                                                                movieId, userId, 
                                                                movie_year,
                                                                genre_score))

#Machine Learning test data using various bias effects and predictions 
#based on them as predictors:

ml_dat_test <- ml_test_set %>% mutate(y = avg_rating) %>% select(-c(rating, 
                                                                    avg_rating,
                                                                    movieId, userId, 
                                                                    movie_year,
                                                                    genre_score))


#Principal Component Analysis of machine learning training data:

pca_4 <- prcomp(ml_dat)

summary(pca_4)
plot(pca_4$sdev, type = "b", xlab = "Principal Component Number")
title(main = "Principal Component Analysis #4")
points(pca_4$sdev, cex = .5, col = "dark red")
points(6 , pca_4$sdev[6], cex = 3, col = "red")
lines(pca_4$sdev, col = "blue")
axis(1, 0:17, col.axis = "blue")
text(x = 6, y = 0.2, labels = "PC6") 
text(x = 6, y = 0.1, labels = "Full Cumulative Proportion Of Variance")

#The first six principal components explain all the variability of the data
#as seen from the summary and the graph:

#Ratings for the algorithms to be trained on:
y <- ml_train_set$avg_rating

#Creating a dataframe with the first six principal components as predictors:
pca_dat <- data.frame(y = ml_train_set$avg_rating, x = pca_4$x[,1:6])


#Using the "caret" package training models to generate predictions and test
#the resulting RMSEs with testing data:

set.seed(1, sample.kind="Rounding")
bayesglm_fit <- train(y ~ ., method = "bayesglm", data = pca_dat)
pred_bayesglm <- predict(bayesglm_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_bayesglm)

set.seed(1, sample.kind="Rounding")
gaussprLinear_fit <- train(y ~ ., method = "gaussprLinear", data = pca_dat)
pred_gaussprLinear <- predict(gaussprLinear_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_gaussprLinear)

set.seed(1, sample.kind="Rounding")
glm_fit <- train(y ~ ., method = "glm", data = pca_dat)
pred_glm <- predict(glm_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_glm)

set.seed(1, sample.kind="Rounding")
glmStepAIC_fit <- train(y ~ ., method = "glmStepAIC", data = pca_dat)
pred_glmStepAIC <- predict(glmStepAIC_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_glmStepAIC)

set.seed(1, sample.kind="Rounding")
glmnet_fit <- train(y ~ ., method = "glmnet", data = pca_dat)
pred_glmnet <- predict(glmnet_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_glmnet)

set.seed(1, sample.kind="Rounding")
pls_fit <- train(y ~ ., method = "pls", data = pca_dat)
pred_pls <- predict(pls_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_pls)

set.seed(1, sample.kind="Rounding")
simpls_fit <- train(y ~ ., method = "simpls", data = pca_dat)
pred_simpls <- predict(simpls_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_simpls)

set.seed(1, sample.kind="Rounding")
kernelpls_fit <- train(y ~ ., method = "kernelpls", data = pca_dat)
pred_kernelpls <- predict(kernelpls_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_kernelpls)

set.seed(1, sample.kind="Rounding")
widekernelpls_fit <- train(y ~ ., method = "widekernelpls", data = pca_dat)
pred_widekernelpls <- predict(widekernelpls_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_widekernelpls)

#Using an ensemble model to to generate predictions and test
#the resulting RMSEs with testing data:

set.seed(1, sample.kind="Rounding")
randomGLM_fit <- train(y ~ ., method = "randomGLM", data = pca_dat)
pred_randomGLM <- predict(randomGLM_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_randomGLM)


#Creating a dataset with average ratings of the same movies as in the sample set,
#taken from the the validation set.

val_samp <- validation %>% filter(movieId %in% ml_test_set$movieId) %>% 
  group_by(movieId) %>% 
  summarise(avg_rating = mean(rating)) %>% pull(avg_rating)

#Calculating the RMSE for each individual model's prediction of average rating  
#for each movie, taken with the actual average rating of that movie in the 
#validation set.

RMSE(val_samp, pred_bayesglm)
RMSE(val_samp, pred_gaussprLinear)
RMSE(val_samp, pred_glm)
RMSE(val_samp, pred_glmStepAIC)
RMSE(val_samp, pred_glmnet)
RMSE(val_samp, pred_pls)
RMSE(val_samp, pred_simpls)
RMSE(val_samp, pred_kernelpls)
RMSE(val_samp, pred_widekernelpls)
RMSE(val_samp, pred_randomGLM)

#Dataframe with all models along with their corresponding RMSE:

final_rmses <- data.frame(Models = c("bayesglm", "gaussprLinear", "glm",
                                     "glmStepAIC", "glmnet", "pls",
                                     "simpls", "kernelpls", "widekernelpls", 
                                     "randomGLM"),
                          ML_RMSE = c(RMSE(val_samp, pred_bayesglm),
                                      RMSE(val_samp, pred_gaussprLinear),
                                      RMSE(val_samp, pred_glm),
                                      RMSE(val_samp, pred_glmStepAIC),
                                      RMSE(val_samp, pred_glmnet),
                                      RMSE(val_samp, pred_pls),
                                      RMSE(val_samp, pred_simpls),
                                      RMSE(val_samp, pred_kernelpls),
                                      RMSE(val_samp, pred_widekernelpls),
                                      RMSE(val_samp, pred_randomGLM)
                          ))

final_rmses_4 <- final_rmses %>% mutate(rmse_rounded = round(ML_RMSE, digits = 3))


#Graph showing RMSEs obtained by each model:

fr_4 <- final_rmses_4 %>% ggplot(aes(Models, ML_RMSE, fill = Models)) + geom_col() +
  geom_hline(yintercept = 0.70, lty = 2, size = 1) +
  geom_text(aes(x = 0, y = 0.70, label = "Average RMSE = 0.70"), nudge_x = 1.4,
            nudge_y = 0.02) +
  geom_text(aes(label = rmse_rounded), nudge_y = 0.05) +
  geom_text(aes(label = signif(ML_RMSE, digits = 6)), 
            angle = 90, nudge_y = -0.4) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
fr_4

#Minimum RMSE obtained:
min(final_rmses_4$ML_RMSE)

#Model yielding the minimal RMSE:
min_ind <- which.min(final_rmses_4$ML_RMSE)
final_rmses_4[min_ind,]

#Average RMSE from all models:
result_4 <- mean(final_rmses_4$ML_RMSE)
result_4


#Sample 5:
#Creating training and test sets for machine learning algorithm training:
set.seed(10000, sample.kind="Rounding")
ml_sample_5 <- edx_pred_dat_l %>% sample_n(1000)

ml_samp_movieids <- ml_sample_5$movieId %>% unique()

ml_rating <- edx_pred_dat_l %>% filter(movieId %in% ml_samp_movieids) %>% 
  group_by(movieId) %>% 
  summarise(avg_rating = mean(rating))

ml_rating_df <- as.data.frame(ml_rating)

ml_sample_set <- inner_join(ml_rating_df, ml_sample_5, by = "movieId")

y <- ml_sample_set$rating

set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)
ml_train_set <- ml_sample_set[-test_index,]
ml_test_set <- ml_sample_set[test_index,]

rm(test_index)

#Machine Learning training data using various bias effects and predictions 
#based on them as predictors:

ml_dat <- ml_train_set %>% mutate(y = avg_rating) %>% select(-c(rating, 
                                                                avg_rating,
                                                                movieId, userId, 
                                                                movie_year,
                                                                genre_score))

#Machine Learning test data using various bias effects and predictions 
#based on them as predictors:

ml_dat_test <- ml_test_set %>% mutate(y = avg_rating) %>% select(-c(rating, 
                                                                    avg_rating,
                                                                    movieId, userId, 
                                                                    movie_year,
                                                                    genre_score))


#Principal Component Analysis of machine learning training data:

pca_5 <- prcomp(ml_dat)

summary(pca_5)
plot(pca_5$sdev, type = "b", xlab = "Principal Component Number")
title(main = "Principal Component Analysis #5")
points(pca_5$sdev, cex = .5, col = "dark red")
points(6 , pca_5$sdev[6], cex = 3, col = "red")
lines(pca_5$sdev, col = "blue")
axis(1, 0:17, col.axis = "blue")
text(x = 6, y = 0.2, labels = "PC6") 
text(x = 6, y = 0.1, labels = "Full Cumulative Proportion Of Variance")

#The first six principal components explain all the variability of the data
#as seen from the summary and the graph:

#Ratings for the algorithms to be trained on:
y <- ml_train_set$avg_rating

#Creating a dataframe with the first six principal components as predictors:
pca_dat <- data.frame(y = ml_train_set$avg_rating, x = pca_5$x[,1:6])


#Using the "caret" package training models to generate predictions and test
#the resulting RMSEs with testing data:

set.seed(1, sample.kind="Rounding")
bayesglm_fit <- train(y ~ ., method = "bayesglm", data = pca_dat)
pred_bayesglm <- predict(bayesglm_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_bayesglm)

set.seed(1, sample.kind="Rounding")
gaussprLinear_fit <- train(y ~ ., method = "gaussprLinear", data = pca_dat)
pred_gaussprLinear <- predict(gaussprLinear_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_gaussprLinear)

set.seed(1, sample.kind="Rounding")
glm_fit <- train(y ~ ., method = "glm", data = pca_dat)
pred_glm <- predict(glm_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_glm)

set.seed(1, sample.kind="Rounding")
glmStepAIC_fit <- train(y ~ ., method = "glmStepAIC", data = pca_dat)
pred_glmStepAIC <- predict(glmStepAIC_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_glmStepAIC)

set.seed(1, sample.kind="Rounding")
glmnet_fit <- train(y ~ ., method = "glmnet", data = pca_dat)
pred_glmnet <- predict(glmnet_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_glmnet)

set.seed(1, sample.kind="Rounding")
pls_fit <- train(y ~ ., method = "pls", data = pca_dat)
pred_pls <- predict(pls_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_pls)

set.seed(1, sample.kind="Rounding")
simpls_fit <- train(y ~ ., method = "simpls", data = pca_dat)
pred_simpls <- predict(simpls_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_simpls)

set.seed(1, sample.kind="Rounding")
kernelpls_fit <- train(y ~ ., method = "kernelpls", data = pca_dat)
pred_kernelpls <- predict(kernelpls_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_kernelpls)

set.seed(1, sample.kind="Rounding")
widekernelpls_fit <- train(y ~ ., method = "widekernelpls", data = pca_dat)
pred_widekernelpls <- predict(widekernelpls_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_widekernelpls)

#Using an ensemble model to to generate predictions and test
#the resulting RMSEs with testing data:

set.seed(1, sample.kind="Rounding")
randomGLM_fit <- train(y ~ ., method = "randomGLM", data = pca_dat)
pred_randomGLM <- predict(randomGLM_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_randomGLM)


#Creating a dataset with average ratings of the same movies as in the sample set,
#taken from the the validation set.

val_samp <- validation %>% filter(movieId %in% ml_test_set$movieId) %>% 
  group_by(movieId) %>% 
  summarise(avg_rating = mean(rating)) %>% pull(avg_rating)

#Calculating the RMSE for each individual model's prediction of average rating  
#for each movie, taken with the actual average rating of that movie in the 
#validation set.

RMSE(val_samp, pred_bayesglm)
RMSE(val_samp, pred_gaussprLinear)
RMSE(val_samp, pred_glm)
RMSE(val_samp, pred_glmStepAIC)
RMSE(val_samp, pred_glmnet)
RMSE(val_samp, pred_pls)
RMSE(val_samp, pred_simpls)
RMSE(val_samp, pred_kernelpls)
RMSE(val_samp, pred_widekernelpls)
RMSE(val_samp, pred_randomGLM)

#Dataframe with all models along with their corresponding RMSE:

final_rmses <- data.frame(Models = c("bayesglm", "gaussprLinear", "glm",
                                     "glmStepAIC", "glmnet", "pls",
                                     "simpls", "kernelpls", "widekernelpls", 
                                     "randomGLM"),
                          ML_RMSE = c(RMSE(val_samp, pred_bayesglm),
                                      RMSE(val_samp, pred_gaussprLinear),
                                      RMSE(val_samp, pred_glm),
                                      RMSE(val_samp, pred_glmStepAIC),
                                      RMSE(val_samp, pred_glmnet),
                                      RMSE(val_samp, pred_pls),
                                      RMSE(val_samp, pred_simpls),
                                      RMSE(val_samp, pred_kernelpls),
                                      RMSE(val_samp, pred_widekernelpls),
                                      RMSE(val_samp, pred_randomGLM)
                          ))

final_rmses_5 <- final_rmses %>% mutate(rmse_rounded = round(ML_RMSE, digits = 3))


#Graph showing RMSEs obtained by each model:

fr_5 <- final_rmses_5 %>% ggplot(aes(Models, ML_RMSE, fill = Models)) + geom_col() +
  geom_hline(yintercept = 0.66, lty = 2, size = 1) +
  geom_text(aes(x = 0, y = 0.66, label = "Average RMSE = 0.66"), nudge_x = 1.4,
            nudge_y = 0.02) +
  geom_text(aes(label = rmse_rounded), nudge_y = 0.05) +
  geom_text(aes(label = signif(ML_RMSE, digits = 6)), 
            angle = 90, nudge_y = -0.4) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
fr_5

#Minimum RMSE obtained:
min(final_rmses_5$ML_RMSE)

#Model yielding the minimal RMSE:
min_ind <- which.min(final_rmses_5$ML_RMSE)
final_rmses_5[min_ind,]

#Average RMSE from all models:
result_5 <- mean(final_rmses_5$ML_RMSE)
result_5


#Sample 6:
#Creating training and test sets for machine learning algorithm training:
set.seed(30000, sample.kind="Rounding")
ml_sample_6 <- edx_pred_dat_l %>% sample_n(1000)

ml_samp_movieids <- ml_sample_6$movieId %>% unique()

ml_rating <- edx_pred_dat_l %>% filter(movieId %in% ml_samp_movieids) %>% 
  group_by(movieId) %>% 
  summarise(avg_rating = mean(rating))

ml_rating_df <- as.data.frame(ml_rating)

ml_sample_set <- inner_join(ml_rating_df, ml_sample_6, by = "movieId")

y <- ml_sample_set$rating

set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)
ml_train_set <- ml_sample_set[-test_index,]
ml_test_set <- ml_sample_set[test_index,]

rm(test_index)

#Machine Learning training data using various bias effects and predictions 
#based on them as predictors:

ml_dat <- ml_train_set %>% mutate(y = avg_rating) %>% select(-c(rating, 
                                                                avg_rating,
                                                                movieId, userId, 
                                                                movie_year,
                                                                genre_score))

#Machine Learning test data using various bias effects and predictions 
#based on them as predictors:

ml_dat_test <- ml_test_set %>% mutate(y = avg_rating) %>% select(-c(rating, 
                                                                    avg_rating,
                                                                    movieId, userId, 
                                                                    movie_year,
                                                                    genre_score))


#Principal Component Analysis of machine learning training data:

pca_6 <- prcomp(ml_dat)

summary(pca_6)
plot(pca_6$sdev, type = "b", xlab = "Principal Component Number")
title(main = "Principal Component Analysis #6")
points(pca_5$sdev, cex = .5, col = "dark red")
points(6 , pca_6$sdev[6], cex = 3, col = "red")
lines(pca_6$sdev, col = "blue")
axis(1, 0:17, col.axis = "blue")
text(x = 6, y = 0.2, labels = "PC6") 
text(x = 6, y = 0.1, labels = "Full Cumulative Proportion Of Variance")

#The first six principal components explain all the variability of the data
#as seen from the summary and the graph:

#Ratings for the algorithms to be trained on:
y <- ml_train_set$avg_rating

#Creating a dataframe with the first six principal components as predictors:
pca_dat <- data.frame(y = ml_train_set$avg_rating, x = pca_6$x[,1:6])


#Using the "caret" package training models to generate predictions and test
#the resulting RMSEs with testing data:

set.seed(1, sample.kind="Rounding")
bayesglm_fit <- train(y ~ ., method = "bayesglm", data = pca_dat)
pred_bayesglm <- predict(bayesglm_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_bayesglm)

set.seed(1, sample.kind="Rounding")
gaussprLinear_fit <- train(y ~ ., method = "gaussprLinear", data = pca_dat)
pred_gaussprLinear <- predict(gaussprLinear_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_gaussprLinear)

set.seed(1, sample.kind="Rounding")
glm_fit <- train(y ~ ., method = "glm", data = pca_dat)
pred_glm <- predict(glm_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_glm)

set.seed(1, sample.kind="Rounding")
glmStepAIC_fit <- train(y ~ ., method = "glmStepAIC", data = pca_dat)
pred_glmStepAIC <- predict(glmStepAIC_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_glmStepAIC)

set.seed(1, sample.kind="Rounding")
glmnet_fit <- train(y ~ ., method = "glmnet", data = pca_dat)
pred_glmnet <- predict(glmnet_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_glmnet)

set.seed(1, sample.kind="Rounding")
pls_fit <- train(y ~ ., method = "pls", data = pca_dat)
pred_pls <- predict(pls_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_pls)

set.seed(1, sample.kind="Rounding")
simpls_fit <- train(y ~ ., method = "simpls", data = pca_dat)
pred_simpls <- predict(simpls_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_simpls)

set.seed(1, sample.kind="Rounding")
kernelpls_fit <- train(y ~ ., method = "kernelpls", data = pca_dat)
pred_kernelpls <- predict(kernelpls_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_kernelpls)

set.seed(1, sample.kind="Rounding")
widekernelpls_fit <- train(y ~ ., method = "widekernelpls", data = pca_dat)
pred_widekernelpls <- predict(widekernelpls_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_widekernelpls)

#Using an ensemble model to to generate predictions and test
#the resulting RMSEs with testing data:

set.seed(1, sample.kind="Rounding")
randomGLM_fit <- train(y ~ ., method = "randomGLM", data = pca_dat)
pred_randomGLM <- predict(randomGLM_fit, ml_dat_test$avg_rating)

RMSE(ml_test_set$avg_rating, pred_randomGLM)


#Creating a dataset with average ratings of the same movies as in the sample set,
#taken from the the validation set.

val_samp <- validation %>% filter(movieId %in% ml_test_set$movieId) %>% 
  group_by(movieId) %>% 
  summarise(avg_rating = mean(rating)) %>% pull(avg_rating)

#Calculating the RMSE for each individual model's prediction of average rating  
#for each movie, taken with the actual average rating of that movie in the 
#validation set.

RMSE(val_samp, pred_bayesglm)
RMSE(val_samp, pred_gaussprLinear)
RMSE(val_samp, pred_glm)
RMSE(val_samp, pred_glmStepAIC)
RMSE(val_samp, pred_glmnet)
RMSE(val_samp, pred_pls)
RMSE(val_samp, pred_simpls)
RMSE(val_samp, pred_kernelpls)
RMSE(val_samp, pred_widekernelpls)
RMSE(val_samp, pred_randomGLM)

#Dataframe with all models along with their corresponding RMSE:

final_rmses <- data.frame(Models = c("bayesglm", "gaussprLinear", "glm",
                                     "glmStepAIC", "glmnet", "pls",
                                     "simpls", "kernelpls", "widekernelpls", 
                                     "randomGLM"),
                          ML_RMSE = c(RMSE(val_samp, pred_bayesglm),
                                      RMSE(val_samp, pred_gaussprLinear),
                                      RMSE(val_samp, pred_glm),
                                      RMSE(val_samp, pred_glmStepAIC),
                                      RMSE(val_samp, pred_glmnet),
                                      RMSE(val_samp, pred_pls),
                                      RMSE(val_samp, pred_simpls),
                                      RMSE(val_samp, pred_kernelpls),
                                      RMSE(val_samp, pred_widekernelpls),
                                      RMSE(val_samp, pred_randomGLM)
                          ))

final_rmses_6 <- final_rmses %>% mutate(rmse_rounded = round(ML_RMSE, digits = 3))


#Graph showing RMSEs obtained by each model:

fr_6 <- final_rmses_6 %>% ggplot(aes(Models, ML_RMSE, fill = Models)) + geom_col() +
  geom_hline(yintercept = 0.68, lty = 2, size = 1) +
  geom_text(aes(x = 0, y = 0.68, label = "Average RMSE = 0.68"), nudge_x = 1.4,
            nudge_y = 0.02) +
  geom_text(aes(label = rmse_rounded), nudge_y = 0.05) +
  geom_text(aes(label = signif(ML_RMSE, digits = 6)), 
            angle = 90, nudge_y = -0.4) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
fr_6

#Minimum RMSE obtained:
min(final_rmses_6$ML_RMSE)

#Model yielding the minimal RMSE:
min_ind <- which.min(final_rmses_6$ML_RMSE)
final_rmses_6[min_ind,]

#Average RMSE from all models:
result_6 <- mean(final_rmses_6$ML_RMSE)
result_6



#Creating a table with results of all random samples combined:

f_rmse_1 <- final_rmses_1 %>% mutate(sample_num = 1L)
f_rmse_2 <- final_rmses_2 %>% mutate(sample_num = 2L)
f_rmse_3 <- final_rmses_3 %>% mutate(sample_num = 3L)
f_rmse_4 <- final_rmses_4 %>% mutate(sample_num = 4L)
f_rmse_5 <- final_rmses_5 %>% mutate(sample_num = 5L)
f_rmse_6 <- final_rmses_6 %>% mutate(sample_num = 6L)

f_rmse_chart <- rbind(f_rmse_1, f_rmse_2, f_rmse_3, f_rmse_4, f_rmse_5, f_rmse_6)
sd_rmse <- sd(f_rmse_chart$ML_RMSE)

#Graph showing results of all sample analyses combined:

f_rmse_chart %>% ggplot(aes(Models, ML_RMSE, fill = Models)) + geom_col() +
  geom_errorbar(aes(ymin = 0.69 - 2*sd_rmse, ymax = 0.69 + 2*sd_rmse)) +
  geom_hline(yintercept = 0.69, lty = 2, size = 1) +
  geom_text(aes(x = 0, y = 0.69, label = "Overall Average RMSE = 0.69"), nudge_x = 6,
            nudge_y = -0.1) +
  geom_text(aes(label = rmse_rounded), angle = 90, nudge_y = -0.4) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  facet_wrap(~sample_num) 

#Minimum RMSE obtained:
min(f_rmse_chart$ML_RMSE)

#Model yielding the minimal RMSE:
min_ind <- which.min(f_rmse_chart$ML_RMSE)
f_rmse_chart[min_ind,]

#Standard Deviation of RMSEs from all models on all samples:
sd_rmse

#Average RMSE from all models:
final_result <- mean(f_rmse_chart$ML_RMSE)
final_result

#Saving data for the RMarkdown file:
edx_head <- head(edx)
validation_head <- head(validation)
edx_pred_dat_head <- head(edx_pred_dat)
edx_pred_dat_head_l <- head(edx_pred_dat_l)

save(edx_head, validation_head, mu, lambda, RMSE, genre_score_chart, 
     G, M, MY, RY, U, edx_pred_dat_head, lambdas, rmses, edx_pred_dat_l_head, 
     final_rmses_1, pca_1, final_rmses_2, pca_2, final_rmses_3, pca_3,
     final_rmses_4, pca_4, final_rmses_5, pca_5, final_rmses_6, pca_6, 
     f_rmse_chart, file = "rmd_dat.RData")
