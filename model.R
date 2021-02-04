#SIMPLE RNN
library(keras)

#laod data anc clean
library(readr)
library(dplyr)
library(jsonlite)
library(stringr)
#review
infile <- "yelp_academic_dataset_review.json"
review_lines <- read_lines(infile, n_max = 200000, progress = TRUE)
reviews_combined <- str_c("[", str_c(review_lines, collapse = ", "), "]")
reviews <- fromJSON(reviews_combined) %>%
  flatten() %>%
  tibble::as_tibble() %>% 
  select(business_id, stars, text)

#business
infile2 <- "yelp_academic_dataset_business.json"
business_lines <- read_lines(infile2, n_max = -1, progress = TRUE)
business_combined <- str_c("[", str_c(business_lines, collapse = ", "), "]")
business <- fromJSON(business_combined) %>%
  flatten() %>%
  tibble::as_tibble() %>% 
  filter(!is.na(categories))
business[["restaurants_binary"]] <- str_detect(business$categories, "Restaurants", negate = F)
business <- business %>% 
  filter(restaurants_binary == TRUE) %>% 
  select(business_id,name,city,stars,review_count,categories, restaurants_binary) 
restaurant_reviews <- reviews %>% 
  filter(business_id %in% business$business_id)
library(textcat)
restaurant_reviews$language <- textcat(restaurant_reviews$text)
restaurant_reviews <- restaurant_reviews %>% filter(language =='english' | language =='scots')
restaurant_reviews <- restaurant_reviews %>%
  mutate(sentiment = case_when(stars <3 ~ 'bad',
                               stars == 3 ~ 'neutral', 
                               stars > 3 ~ 'good') #T is means in all other cases, practically.
  )
review <-  restaurant_reviews %>%
  select(text, sentiment)


#now we could stuck more rnn layer, but there must be the return sequennce = TRUE

#1 --> preprocess raw data

good_reviews <- review %>%
  filter(sentiment=='good') %>% 
  select(text,sentiment)
head(good_reviews)
nrow(good_reviews)
bad_reviews <- review %>%
  filter(sentiment=='bad') %>% 
  select(text,sentiment)
head(bad_reviews)
nrow(bad_reviews)

review <- rbind(good_reviews[1:25000,],bad_reviews[1:25000,],good_reviews[25001:30000,])

#prepare train data
data1 <- review %>% 
  select(text,sentiment) %>% 
  mutate(label = ifelse(sentiment=='good',1,0))
head(data1)
nrow(data1)
sum(data1$label)
#now I have a balance training set

text <- as.vector(data1$text)   
label <- as.vector(data1$label)
length(label)
#clean data
library(tm)
library(stopwords)
library(hunspell)
library(textstem)

dfCorpus <- Corpus(VectorSource(text))
clean.corpus <- function(corpus){
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeWords, stopwords('en'))
  corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, content_transformer(lemmatize_strings))
  return(corpus)
}
dfCorpus <-  clean.corpus(dfCorpus)

df <- data.frame(text = sapply(dfCorpus, function(x) x),
                 stringsAsFactors = F)
head(df)

text <- df$text
head(text)

length(text)

#2 --> tokenization

training_samples <- 40000
test_samples  <- 15000
max_feature <- 20000
maxlen <- 50
batch_size <- 32

tokenizer <- text_tokenizer(num_words = max_feature) %>%
  fit_text_tokenizer(text)
sequences <- texts_to_sequences(tokenizer, text)
word_index = tokenizer$word_index

cat("Found", length(word_index), "unique tokens.\n")
data <- pad_sequences(sequences, maxlen = maxlen)

#plit in test,validation
indices <- sample(1:50000)
training_indices <- indices[1:40000]
test_indices <- indices[40001:50000]
x_train <- data[training_indices,]
y_train <- label[training_indices]
x_test <- data[test_indices,]
y_test <- label[test_indices]
sum(y_train)
sum(y_test)
x_test <- rbind(x_test,data[50001:55000,])
y_test <- c(y_test,label[50001:55000])
sum(y_test)
length(y_test)

#reshaffle test set
test_indices <- sample(1:15000)
x_test <- x_test[test_indices,]
y_test <- y_test[test_indices]
#3 --> model
#model 1

model1 <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_feature, output_dim = 100,
                  input_length = maxlen) %>%
  layer_flatten() %>%
  layer_dropout(0.2) %>% 
  layer_dense(units = 100, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid") %>% 
  compile(optimizer=optimizer_rmsprop(lr = 0.001,rho = 0.9), loss= 'binary_crossentropy', metrics   =  c('accuracy') )


history1 <- model1 %>% fit(
  x_train, y_train,
  epochs = 3,
  batch_size = 32,
  validation_split = 0.3
)
plot(history1)
history1

result1 <-model1 %>% evaluate(x_test,y_test)

#??? model2
model2 <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_feature,output_dim = 100,input_length = maxlen) %>% 
  layer_simple_rnn(units=100,dropout = 0.2) %>% 
  layer_dense(units = 1,activation='sigmoid')


model2 %>%  compile(
  optimizer = 'rmsprop',
  loss='binary_crossentropy',
  metric = c('acc')
)

history2 <- model2 %>% fit(
  x_train,y_train,
  epochs = 3,
  batch_size = 32,
  validation_split = 0.3
)
plot(history2)
history2
result2 <-model2 %>% evaluate(x_test,y_test)


 
#model 3

model3 <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_feature,output_dim = 100,input_length = maxlen) %>% 
  layer_lstm(units = 100, dropout = 0.2,recurrent_dropout = 0.2) %>% 
  layer_dense(units = 1,activation='sigmoid')


model3 %>%  compile(
  optimizer = 'rmsprop',
  loss='binary_crossentropy',
  metric = c('acc')
)

history3 <- model3 %>% fit(
  x_train,y_train,
  epochs = 4,
  batch_size = 32,
  validation_split = 0.3
)
plot(history3)

result3 <-model3 %>% evaluate(x_test,y_test)
history3$metrics
#conv nets

model4 <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_feature,output_dim = 100,input_length = 50) %>% 
  layer_dropout(0.2) %>% 
  layer_conv_1d(filters = 64, kernel_size = 5, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 4) %>%
  layer_lstm(units = 100, dropout = 0.2,recurrent_dropout = 0.2) %>% 
  layer_dense(units = 1,activation='sigmoid')


model4 %>%  compile(
  optimizer = 'rmsprop',
  loss='binary_crossentropy',
  metric = c('acc')
)

history4 <- model4 %>% fit(
  x_train,y_train,
  epochs = 4,
  batch_size = 32,
  validation_split = 0.3
)
plot(history4)

#evaluation

result1 <-model1 %>% evaluate(x_test,y_test)
history1

result2 <-model2 %>% evaluate(x_test,y_test)
history2

result3 <-model3 %>% evaluate(x_test,y_test)
history3$metrics

result4 <-model4 %>% evaluate(x_test,y_test)
history4$metric

