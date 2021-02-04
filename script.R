
library(readr)
library(dplyr)
library(jsonlite)
library(stringr)

############################reviews_dataset
infile <- "yelp_academic_dataset_review.json"
review_lines <- read_lines(infile, n_max = 20000, progress = TRUE)

reviews_combined <- str_c("[", str_c(review_lines, collapse = ", "), "]")

reviews <- fromJSON(reviews_combined) %>%
  flatten() %>%
  tibble::as_tibble() %>% 
  select(business_id, stars, text)

reviews <- reviews %>% 
  select(business_id, stars, text) 
head(reviews)

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
  select(business_id,name,stars) 
head(business)

restaurant_reviews <- reviews %>% 
  filter(business_id %in% business$business_id)
head(restaurant_reviews)

library(ggplot2)

ggplot(restaurant_reviews, aes(x=stars))+
  geom_bar(stat="bin", bins= 9, fill="steelblue") + 
  geom_text(stat='count', aes(label=..count..), vjust=1.9, color="white") +
  ggtitle("frequency of stars") +
  xlab("Stars") + ylab("Count") +
  theme_minimal()

######## back to text mining

library(textcat)
#keep only english reviews
restaurant_reviews$language <- textcat(restaurant_reviews$text)

ggplot(restaurant_reviews, aes(x=language))+
  geom_bar(stat="count", fill="steelblue") + 
  ggtitle("Language Count") +
  xlab("Language") + ylab("Count") +
  theme_minimal() + coord_flip()

#we have keep only the english sentences
restaurant_reviews <- restaurant_reviews %>% filter(language =='english' | language =='scots')

restaurant_reviews <- restaurant_reviews %>%
  mutate(sentiment = case_when(stars <3 ~ 'bad',
                               stars == 3 ~ 'neutral', 
                               stars > 3 ~ 'good') #T is means in all other cases, practically.
  )


# Libraries
library(tidyverse)
library(tidytext)
library(tm)
library(stopwords)
library(hunspell)
library(textstem)

doc_rev<- restaurant_reviews %>%
  select(business_id,text, sentiment)

docs_long <- doc_rev %>% 
  unnest_tokens(word, # name of the column with single tokens in output
                text)  # name of the column with docs in input
docs_long
nrow(docs_long)


docs_long <- doc_rev %>% 
  unnest_tokens(word, text) %>% 
  filter(str_length(word) > 1) %>% 
  filter(!str_detect(word, '[0-9]')) %>% 
  anti_join(stop_words, by = 'word')  

nrow(docs_long)

#lemmization
docs_long <- docs_long %>%
  mutate(word2 = lemmatize_words(word)) %>% 
  mutate(word2 = sapply(word2, function(x) x[1])) # simplifying

#Imputing original values for NA's
docs_long <- docs_long %>% 
  mutate(word2 = ifelse(is.na(word2), word, word2)) %>% 
  filter(!word2=='food')

word_tokens <- docs_long %>% 
  select(word2,sentiment) 
     
#################### WORD CLOUD #######################################################à

word_count <- word_tokens %>% 
  count(word2) 


#PLOT 1 WORDCLOUD
windows() # adjusting margins
wordcloud(word_count$word2, 
          word_count$n, max.words = 80, 
          rot.per = 0.35, colors = brewer.pal(6, "Dark2"))
#PLOT 3 WORDCLOUD

word_count <- word_tokens %>% 
  count(sentiment,word2) 

par(mar = c(0, 0, 0, 0), mfrow = c(1,3))
for (i in unique(word_count$sentiment)){
  word_sentiment <- filter(word_count, sentiment == i)
  wordcloud(word_sentiment$word2, word_sentiment$n, max.words = 100, 
            rot.per = 0.35, colors = brewer.pal(6, "Dark2"))}


#COMPARISON PLOT
  
#USING TF  
word_top <- word_tokens %>% 
    count(sentiment, word2, sort = T) %>% 
    filter(!word2 %in% c('food','restaurant')) %>%
    bind_tf_idf(word2, sentiment, n) %>%
    group_by(sentiment) %>% 
    top_n(15, tf) %>% 
    ungroup() %>% 
    mutate(word = reorder(word2, tf))
windows()
ggplot(word_top, aes(x = reorder_within(word, tf, sentiment), y = tf, fill =sentiment)) +
    geom_col(show.legend = F) +
    coord_flip() +
    facet_wrap(~ sentiment, scales ="free_y" ) +
    scale_x_reordered() #reordering with

#COMMONALITY/COMPARISON PLOT


word_count_good <- word_tokens %>% 
  filter(sentiment=='good') %>% 
  group_by(word2,sentiment) %>% 
  count() %>% 
  bind_tf_idf(word2, sentiment, n) %>% 
  select(word2,sentiment,n)

word_count_bad <- word_tokens %>% 
  filter(sentiment=='bad') %>% 
  group_by(word2,sentiment) %>% 
  count() %>% 
  bind_tf_idf(word2, sentiment, n) %>% 
  select(word2,sentiment,n)

corpora <- bind_rows(word_count_good, word_count_bad) %>% 
  spread(sentiment, n, fill = 0) %>% 
  column_to_rownames("word2") %>% 
  as.matrix()

# We can choose pallete for commonality and comparison clouds:
display.brewer.all()
pal <- brewer.pal(8, "Greens")
pal <- pal[-(1:4)]
par(mar = c(0, 0, 0, 0), mfrow = c(1,2))
commonality.cloud(corpora, 
                  max.words = 200, 
                  random.order = FALSE,
                  colors = pal)


comparison.cloud(corpora, 
                 max.words = 200, 
                 random.order = FALSE, 
                 title.size = 1.0, 
                 colors = brewer.pal(ncol(corpora),'Set1'))
#TAKE ALL COMMON TERM IN 1 TABLE
common.terms <- subset(corpora, corpora[, 1] > 0 & corpora[, 2] > 0)

# Look at the word that has most difference in time use(to do so I should use the same number of bad and good reviews)
diff <- abs(common.terms[, 1] - common.terms[, 2])
common.terms <- cbind(common.terms, diff)
common.terms <- common.terms[order(common.terms[, 3], decreasing = T), ]

head(common.terms)
tail(common.terms)

# First 25 obs: 
df <- data.frame(x = common.terms[1:25, 1], 
                 y = common.terms[1:25, 2],
                 labels = rownames(common.terms)[1:25]
)


# Palette:
colours <- colorRampPalette(brewer.pal(9,'Greens'))(25)
colours <- rev(colours)
windows()
# Visualisation:
pyramid.plot(df$x, df$y, 
             labels = df$labels, 
             gap = 100, 
             top.labels = c("BAD", "terms", "GOOD"), # labels
             main = "terms in Common", # tutle
             lxcol = colours, rxcol = colours, # colors
             laxlab = NULL, raxlab = NULL, unit = NULL)


#BI_GRAMS
library(widyr)
library(igraph)
library(ggraph)
library(janeaustenr)
library(gutenbergr)

restaurant_reviews <- restaurant_reviews %>% 
  select(text,sentiment)
restaurant_reviews %>%    
  count(sentiment)

bigrams <- restaurant_reviews %>% 
  unnest_tokens(bigram, text, token = 'ngrams', n = 2)

new_bigrams <- bigrams %>% 
  separate(bigram, c('c1', 'c2'), sep = " " ) # splitting bigrams into 2 columns

# Now we will remove stop words from the analysis
filtered_bigrams <- new_bigrams %>% 
  filter(!c1 %in% stop_words$word) %>% 
  filter(!c2 %in% stop_words$word)

# Let's count the most frequent bigrams again:
bigram_counts <- filtered_bigrams %>% 
  count(c1, c2, sort = TRUE)
bigram_counts

# Now we can combine the columns back into one:
bigrams_united <- filtered_bigrams %>% 
  unite(bigram, c1, c2, sep = " ") # pastes together multiple columns into one
bigrams_united # filtered bigrams 

filtered_bigrams %>% # because we will analyze the separated columns without the stop words
  filter(c1 == 'service') %>%
  count(sentiment, c2, sort = TRUE) 

filtered_bigrams %>% # because we will analyze the separated columns without the stop words
  filter(c2 == 'service') %>%
  count(sentiment, c1, sort = TRUE) 

filtered_bigrams %>% # because we will analyze the separated columns without the stop words
  filter(c1 == 'food') %>% 
  count(sentiment, c2, sort = TRUE) 

filtered_bigrams %>% # because we will analyze the separated columns without the stop words
  filter(c2 == 'food') %>% 
  count(sentiment, c1, sort = TRUE) 



#for each type
bigram_tfidf <- bigrams_united %>% # we analyze bigrams without stopwords
  count(sentiment,bigram) %>% # how many times a particular bigram occured in the book
  bind_tf_idf(bigram, sentiment, n) %>% 
  arrange(desc(tf_idf)) 
bigram_tfidf


# We will now visualize bigrams with the highest tf_idf for each book:
bigram_top8<- bigram_tfidf %>% 
  arrange(desc(tf_idf)) %>% # first we have to arrange them in the descending order
  mutate(bigram = factor(bigram, levels = rev(unique(bigram)))) %>% # a new variable, to order values 
  group_by(sentiment) %>% 
  top_n(8) %>% # top 8 results
  ungroup


ggplot(data = bigram_top8, aes(bigram, tf_idf, fill = sentiment)) +  
  geom_col(show.legend = FALSE) +
  labs(x = NULL, y = NULL) + 
  facet_wrap(~sentiment, ncol = 3, scales = "free_y") + 
  coord_flip()

