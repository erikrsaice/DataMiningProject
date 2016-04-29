install.packages("gbm")
install.packages("Metrics")
install.packages("readr")
install.packages("SnowballC")
install.packages("xgboost")

#load packages from the library
library(gbm)
library(Metrics)
library(readr)
library(SnowballC)
library(xgboost)

# read data into variable and store in the environment
# change path of read_csv to where your files are stored
cat("Reading data\n")
train <- read_csv('C:/Users/Erik/Desktop/DataMining/Project/train.csv')
test <- read_csv('C:/Users/Erik/Desktop/DataMining/Project/test.csv')
desc <- read_csv('C:/Users/Erik/Desktop/DataMining/Project/product_descriptions.csv')

# This will merge the product descriptions into the train and test data for comparison in tests
# Think of it like a join in databases
cat("Merge description with train and test data \n")
train <- merge(train,desc, by.x = "product_uid", by.y = "product_uid", all.x = TRUE, all.y = FALSE)
test <- merge(test,desc, by.x = "product_uid", by.y = "product_uid", all.x = TRUE, all.y = FALSE)

# BEGIN FUNCTIONS
# The functions are breaking the strings into tokens for comparison using space as a delimiter
# These will be called and executed down below
t <- Sys.time()
word_match <- function(words,title,desc){
  n_title <- 0
  n_desc <- 0
  words <- unlist(strsplit(words," "))
  nwords <- length(words)
  for(i in 1:length(words)){
    pattern <- paste("(^| )",words[i],"($| )",sep="")
    n_title <- n_title + grepl(pattern,title,perl=TRUE,ignore.case=TRUE)
    n_desc <- n_desc + grepl(pattern,desc,perl=TRUE,ignore.case=TRUE)
  }
  return(c(n_title,nwords,n_desc))
}
word_match2 <- function(words,title,desc){
  n_title <- 0
  n_desc <- 0
  words <- unlist(strsplit(words," "))
  nwords <- length(words)
  for(i in 1:length(words)){
    pattern <- words[i]
    n_title <- n_title + grepl(pattern,title,ignore.case=TRUE)
    n_desc <- n_desc + grepl(pattern,desc,ignore.case=TRUE)
  }
  return(c(n_title,nwords,n_desc))
}
word_match3 <- function(words,title,desc){
  n_title <- 0
  n_desc <- 0
  words <- unlist(strsplit(words," "))
  nwords <- length(words)
  for(i in 1:length(words)){
    pattern <- paste("[0-","9]",sep="")
    n_title <- n_title + grepl(pattern,title,perl=TRUE,ignore.case=TRUE)
    n_desc <- n_desc + grepl(pattern,desc,perl=TRUE,ignore.case=TRUE)
  }
  return(c(n_title,nwords,n_desc))
}

#This handles stemming of the words
word_stem <- function(words){
  
  i <- 1
  words <- unlist(strsplit(words," "))
  nwords <- length(words)
  pattern <- wordStem(words[i], language = "porter")
  for(i in 2:length(words)){
    pattern <- paste(pattern,wordStem(words[i], language = "porter"),sep=" ")
  }
  return(pattern)
}

#the functions will end up comparing whole words, numbers, and stemmed versions of words

#END FUNCTIONS

# use function word_match to get a count of how many words are matching for training/test set
# 3 new columns will be created in train/test to hold the counts called nmatch_title, nwords, nmatch_desc
# nmatch_title is number of words in user search matching title
# nwords is number of words in user search
# nmatch_desc is number of words in user search matching product description

cat("Get number of words and word matching title in train\n")
train_words <- as.data.frame(t(mapply(word_match,train$search_term,train$product_title,train$product_description)))
train$nmatch_title <- train_words[,1]
train$nwords <- train_words[,2]
train$nmatch_desc <- train_words[,3]

cat("Get number of words and word matching title in test\n")
test_words <- as.data.frame(t(mapply(word_match,test$search_term,test$product_title,test$product_description)))
test$nmatch_title <- test_words[,1]
test$nwords <- test_words[,2]
test$nmatch_desc <- test_words[,3]

#removes variables from environment containing counts
rm(train_words,test_words)

#porter stem count coincidences
#now words will be stemmed and compared
#counts will be stored in separate columns
cat("Get number of words and word matching title in train with porter stem\n")
train$search_term2 <- sapply(train$search_term,word_stem)
train_words <- as.data.frame(t(mapply(word_match2,train$search_term2,train$product_title,train$product_description)))
train$nmatch_title2 <- train_words[,1]
train$nmatch_desc2 <- train_words[,3]
train$search_term2 <- sapply(train$search_term,word_stem)
train_words <- as.data.frame(t(mapply(word_match3,train$search_term2,train$product_title,train$product_description)))
train$nmatch_title3 <- train_words[,1]
train$nmatch_desc3 <- train_words[,3]
summary(train)
train$search_term2 <-NULL

#porter stem count coincidences
cat("Get number of words and word matching title in test with porter stem\n")
test$search_term2 <- sapply(test$search_term,word_stem)
train_words <- as.data.frame(t(mapply(word_match2,test$search_term2,test$product_title,test$product_description)))
test$nmatch_title2 <- train_words[,1]
test$nmatch_desc2 <- train_words[,3]
train_words <- as.data.frame(t(mapply(word_match3,test$search_term2,test$product_title,test$product_description)))
test$nmatch_title3 <- train_words[,1]
test$nmatch_desc3 <- train_words[,3]
test$search_term2 <-NULL
rm(test_words)

#training the model here
#click on xgboost in the packages to see what all the parameters do, I don't understand all of them
#you will get different rmse every time it it run
#uses columns 7 through 13 in training data

cat("A simple linear model on number of words and number of words that match\n")
h<-sample(nrow(train),10000)
dval<-xgb.DMatrix(data=data.matrix(train[h,7:13]),label=train[h,5])
dtrain<-xgb.DMatrix(data=data.matrix(train[-h,7:13]),label=train[-h,5])
watchlist<-list(eval=dval,train=dtrain)
param <- list(  objective           = "reg:linear", 
                booster             = "gbtree",
                eta                 = 0.025, 
                max_depth           = 6, 
                subsample           = 0.6, 
                colsample_bytree    = 1, 
                eval_metric         = "rmse",
                min_child_weight    = 8
)



clf <- xgb.train(data                 = dtrain, 
                 params               = param, 
                 nrounds              = 1000, #300
                 verbose              = 1,#2
                 watchlist            = watchlist,
                 early.stop.round     = 50, 
                 print.every.n        = 1
)
clf$bestScore





# predict relevance scores of test data
# this will use columns 6 through 12 in test data

cat("Submit file\n")
test_relevance <- predict(clf,data.matrix(test[,6:12]),ntreelimit =clf$bestInd)
summary(test_relevance)
#simple if else statement to keep the relevance values within our domain of 1 and 3
test_relevance <- ifelse(test_relevance>3,3,test_relevance)
test_relevance <- ifelse(test_relevance<1,1,test_relevance)


#submission <- data.frame(id=test$id,relevance=test_relevance)
#write_csv(submission,"xgb_submission.csv")
print(Sys.time()-t)

