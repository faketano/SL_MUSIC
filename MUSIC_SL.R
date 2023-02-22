# Classification Multiclass. Statistical Learning Salini 2021/2022
# Libraries 
library(rpart)
library(rpart.plot)
library(party)
library(partykit)
library(dplyr)
library(tidyverse)
library(tree)
library(Metrics)
library(caret)
library(tree)
library(ISLR)
library(MASS)
library(plyr)
library(purrr)
library(rsample)
library(ipred)
library(class)
library(doSNOW)
library(foreach)
library(rattle)
library(e1071)
library(dplyr)
library(rpart)
library(rpart.plot)
library(Metrics)
library(mlr)
library(ggplot2)
library(plotly)
library(caret)
library(nnet)
library(class)
library(ggplot2)
library(plyr)
require(gridExtra)
library(caret)
library(ISLR)
library(class)
library(MASS)
library(party)
library(tree)
library(rattle)
library(rsample)
library(lares)
library(corrplot)
library(recipes)
library(parsnip)
library(workflows)
library(ggplot2)
library(pacman)
library(GGally)


df <- read.csv("/Users/matiasluraschi/Desktop/SL Project/archive (1) 2/music.csv")
dim(df) #WE HAVE 15 FEATURES AND 17996 OBSERVATIONS

#Now that we have our dataset and first doing the Exploratory Data Analysis
#Let's see if we have any cells that might be disrupting. Here we check if we have NA values
sum(is.na(df)) #WE HAVE 6819 NA VALUES. WE NEED TO TREAT THEM IN ORDER TO USE A MODEL

#WHEN WE VIEW OUR DATASET, WE SEE THAT MOST OF OUR VALUES ARE NORMALIZED.
# WE ARE GOING TO DROP THE NA VALUES THAT WE HAVE. 
# WE ARE GOING TO USE DIFFERENT METHODS TO SHOW HOW THEY MIGHT AFFECT LATER OUR ANALYSIS

# 1) DROPPING ROWS WITH NA VALUES 
#We are going to DROP the ROWS with the NA VALUES, but we are going to loose 6819 observations
#DROP THE ROWS WITH NA VALUES
df <- df %>% drop_na()
dim(df) #11813 rows and 15 features
________________________________________________________________________________

# Now we are going to do some exploratory data analysis
# to see if we can get a feel of our data 
# We are going to do this analysis by grouping on the
# class variables that have been pre-coded. 
# They go from 1 to 10 each meaning a different musical genre. 

# LUCKILY MOST OF OUR FEATURES RANGE FROM 0 TO 1
# WE ARE GOING TO NORMALIZE THREE OF THEM THAT ARE NOT WITHIN THAT RANGE
# WE DO SO WE DON'T CONFUSE THE ALGORITHM BY GIVING IT DIFFERENT MAGNITUDES
#WE NORMALIZE POPULARITY, LOUDNESS AND TEMPO.
preproc <- preProcess(df[,c(1,5,12)], method=c("range"))
norm <- predict(preproc, df[,c(1,5,12)])
dim(df)
summary(norm) # HERE I HAVE THE 3 STANDARIZED VARIABLES.
# NOW I JUST HAVE TO APPEND THEM TO THE OTHER DATASET AND DELETE THE ORIGINAL ONES
dim(norm)
dim(df)

#add columns called 'new1' and 'new2'
df_total <- cbind(df, norm$Popularity,norm$loudness,norm$tempo)
view(df_total)

#DROPPING COLUMNS THAT ARE NOT NORMALIZED
df_total$Popularity <- NULL
df_total$loudness <- NULL
df_total$tempo <- NULL
view(df_total)

#NOW ALL MY VARIABLES ARE NORMALIZED BUT MINUTES AND SOME OTHERS THAT NEED TO HAVE A SCALE

______________________________________________________________________________


view(df_total)
# ELIMINATING OUTLIERS USING Z SCORE

#find absolute value of z-score for each value in each column
z_scores <- as.data.frame(sapply(df_total, function(df_total)(abs(df_total-mean(df_total))/sd(df_total))))
#view first six rows of z_scores data frame
head(z_scores)
#ADD THE ORIGINAL CLASS AND MINUTES
z_scores1 <- cbind(z_scores, df_total$Class)
z_scores1 <- cbind(z_scores1, df_total$MINUTES)
view(z_scores1)
dim(z_scores1)
#DROP
z_scores1$Class <- NULL #DROP CLASS OLD
z_scores1$MINUTES <- NULL #DROP MINUTES OLD

names(z_scores1)[names(z_scores1) == 'norm$loudness'] <- 'Loudness' #RENAMING loudness
names(z_scores1)[names(z_scores1) == 'norm$tempo'] <- 'Tempo' #RENAMING Tempo
names(z_scores1)[names(z_scores1) == 'norm$Popularity'] <- 'Popularity' #RENAMING Popularity

#NOW I HAVE A FIXED DATASET ON THE VARIABLE Z_SCORES1 and I need to do the condition


#only keep rows in dataframe with all z-scores less than absolute value of 3 
no_outliers <- z_scores1[!rowSums(z_scores1[,c('danceability','energy','key','mode','speechiness','acousticness','instrumentalness','liveness','valence','time_signature','Popularity','Loudness','Tempo')]>3), ]

#view row and column count of new data frame
dim(no_outliers)
# WHEN I ELIMINATED MY OUTLIERS. I DROPED A LITTLE MORE THAN A THOUSAND. 
# MY FINAL DATASET NOW HAS 10773 ROWS AND 15 FEATURES
view(no_outliers)
#IM GONNA SAVE THIS 
write.csv(no_outliers,"/Users/matiasluraschi/Desktop/SL Project/final_df_NO.csv", row.names=FALSE)

________________________________________________________________________________
________________________________________________________________________________
________________________________________________________________________________

#NOW I HAVE MY FINAL DATASET. I'M CALLING IT df_f

df_f <- read_csv("/Users/matiasluraschi/Desktop/SL Project/final_df_NO.csv")
str(df_f)
dim(df_f)


#df <- transform(df, mode = as.numeric(mode))
#df <- transform(df, time_signature = as.numeric(time_signature))
#df <- transform(df, Class = as.numeric(Class))

# Exploratory Data Analysis

# Boxplot Popularity
boxplot(df_f$Popularity ~ df_f$Class, col=rainbow(11))

# In this boxplot we see that even though the popularity of the different 
# cathegories are somewhat different, 
# we can not tell that they're statistically different because the boxplots are overlapping.


# Boxplot Danceability
boxplot(df_f$danceability ~ df_f$Class, col=rainbow(11))
# Here we can tell that the most "danceable" cathegory is cathegory 5

# BOXPLOT VALENCE = WHEN A SONG IS "HAPPY"
boxplot(df_f$valence ~ df_f$Class, col=rainbow(11))
# Valence is a measure of happines. Here, the less happy cathegory is 7.  

# BOXPLOT MINUTI
boxplot(df_f$Minutes ~ df_f$Class, ylim =c(0,10), col=rainbow(11))
# We can tell here that all the cathegories seem to have more or less the same duration.

# ______________________________________________________________

# We use the following code to find out how many songs of each category. 
counting <- count(df_f, "Class")
# We see that it is clearly unbalanced. At the time of applying a model, if we were not to take
# this into account, we would have biased results. 
max(counting$freq) #3227
min(counting$freq) #146

# We have to balance this dataset. HOW ARE WE GONNA DO IT? WE USE SMOTE. 
# WE DO IT WITH PYTHON BECAUSE IT IS A MESS TO DO IT WITH R
barplot(prop.table(table(df_f$Class)),
        col = rainbow(10),
        ylim = c(0, 0.3),
        main = "Original Class Distribution")
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
# Importing balanced Dataset on Python
df_b <- read.csv("/Users/matiasluraschi/Desktop/SL Project/df_f_balanced.csv")
view(df_b)
dim(df_b)
df_b <- df_b[-1]
df_b$Class <- as.factor(df_b$Class)
view(df_b)

# let's see its distribution. We see that it is balanced so we can use it to train our model

barplot(prop.table(table(df_b$Class)),
        col = rainbow(10),
        ylim = c(0, 0.12),
        main = "Class Distribution",
        )

# --------------------------------------------------
#ENCODING 
df_b$Class <- as.numeric(as.factor(df_b$Class)) # knn

#TRAIN AND TEST DATASET
# 70% of the sample size
smp_size <- floor(0.70 * nrow(df_b))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(df_b)), size = smp_size)

train <- df_b[train_ind, ]
test <- df_b[-train_ind, ]

#------------------------------------------------------
#------------------------------------------------------
# PRINCIPAL COMPONENT ANALYSIS

# WE ARE GOING TO TRY TO ELIMINATE FEATURES THAT ARE NOT RELEVANT 
# IN ORDER TO USE PCA, IT'S BEST IF OUR VARIABLES ARE CORRELATED
# WE CHECK THAT WITH A CORRELATION MATRIX, 

CORRELATION <-cor(df_b)
corrplot(CORRELATION, method="number",shade.col='BLACK',order='FPC' ,tl.srt=45,tl.cex=0.75,number.cex=0.75,is.corr=TRUE,title='Correlation Matrix - Music Features')

#PCA should be used mainly for variables which are strongly correlated.
# If the relationship is weak between variables, PCA does not work well to reduce data.
#In general, if most of the correlation coefficients are smaller than 0.3, PCA will not help.

# We analize our data and we decide that it won't be useful to use PCA. However, let's compute it just to see.


#PCA
df_recipe <- recipe(~ ., data = df_b) %>% 
  update_role(Class, new_role = "id") %>%  
  # step_naomit(all_predictors()) %>% 
  step_normalize(all_predictors()) %>% 
  step_pca(all_predictors(), id = "pca")

df_recipe

#DF PREPARATION

df_prep <- prep(df_recipe)
df_prep

#PCA

tidy_pca_loadings <- df_prep %>% 
  tidy(id = "pca")
tidy_pca_loadings

#BAKING

df_bake <- bake(df_prep, df_b)
df_bake  # has the PCA LOADING VECTORS that we are familiar 


#CHECKING NUMBER OF PC
df_prep$steps[[2]]$res$sdev

# VARIANCE EXPLAINED
df_prep %>% 
  tidy(id = "pca", type = "variance") %>% 
  filter(terms ==  "percent variance") %>% 
  ggplot(aes(x = component, y = value)) +
  geom_point(size = 2) +
  geom_line(size = 1) +
  scale_x_continuous(breaks = 1:4) +
  labs(title = "% Variance explained",
       y = "% total variance",
       x = "PC",
       caption = "Source: ChemometricswithR book") +
  geom_text(aes(label = round(value, 2)), vjust = -0.3, size = 4) +
  theme_minimal() +
  theme(axis.title = element_text(face = "bold", size = 12),
        axis.text = element_text(size = 10),
        plot.title = element_text(size = 14, face = "bold"))  # 2 or 3


# CUMULATIVE VARIANCE PLOT
df_prep %>% 
  tidy(id = "pca", type = "variance") %>% 
  filter(terms == "cumulative percent variance") %>%
  ggplot(aes(component, value)) +
  geom_col(fill= rainbow(13)) +
  labs(x = "Principal Components", 
       y = "Cumulative variance explained (%)",
       title = "Cumulative Variance explained") +
  geom_text(aes(label = round(value, 2)), vjust = -0.2, size = 4) +
  theme_minimal() +
  theme(axis.title = element_text(face = "bold", size = 12),
        axis.text = element_text(size = 10),
        plot.title = element_text(size = 14, face = "bold")) 


# SINCE LESS THAN 30% OF THE VARIABILITY CAN BE EXPLAINED BY THE FIRST 2 PC
# AND LESS THAN 40% WITH THE FIRST THREE, WE DECIDED NOT TO DO IT.
#WE USE OUR NORMAL DATASET OTHERWISE CONCLUTIONS COULD BE WRONG


------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------
#TREE
#Fit the tree model to the training data. 
# More likely already includes pruning in the algorithm. GOOGLEAR! 

tree <-rpart(Class ~ ., data=train, method="class")
printcp(tree)

# Variables actually used in tree construction:
#[1] acousticness     energy           instrumentalness Minutes         
#[5] mode             Popularity       speechiness 

#WE SEE THAT HIGHEST CORRELATIONS WERE BETWEEN 
#energy, loudness, acousticness, instrumentalness, (+)
# popularity, speechiness, danceability

#IT SEEMS THAT THE VARIABLE WITH THE HIGHEST CORRELATION ARE CONSIDERED IN THE TREE


tree_predict <- predict(tree, test, type="class")
confusionMatrix(data=tree_predict, reference = test$Class)
accuracy(test$Class, tree_predict) #0.3750 DI ACCURACY
#THIS TREE HAS ALREADY Hyperparametrization Tuning



# Tree with Information Criteria 
tree_information <-rpart(Class ~ ., 
                         train,
                         method="class", 
                         parms=list(split="information"))

tree_information_predict <- predict(tree_information, test, type="class")

confusionMatrix(data=tree_information_predict, reference = test$Class)
accuracy(test$Class, tree_information_predict) #0,35 di accuracy

#Hiper Parametization

best <- tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]
tree_pruned <- prune(tree, cp = best)
predict_pruned <- predict(tree_pruned, test, type="class")
accuracy(test$Class, predict_pruned)  #Accuracy is 0,37, has gone up 2% by pruning


-----------------------------------------------------------------------------------------
  
# CROSS VALIDATION ON THE TREE - GOOGLE WHY CROSSVALIDATION GETS ACCURACY HIGHER
ctrl  <- trainControl(method  = "cv",number  = 10) #, summaryFunction = multiClassSummary
fit.cv <- train(Class ~ ., data = train, method = "rpart",
                trControl = ctrl, 
                #preProcess = c("center","scale"), 
                #tuneGrid =data.frame(cp=0.05))
                tuneLength = 30) # metric="Kappa",

pred <- predict(fit.cv,test)
confusionMatrix(table(test[,"Class"],pred))
accuracy(test$Class, pred) #ACCURACY OF THE TREE USING CROSS-VALIDATION 0,44
print(fit.cv)
plot(fit.cv)
plot(fit.cv$finalModel)
text(fit.cv$finalModel)

--------------------------------------------------------------------------------------
#CODICE PER FARE UN ALBERO CARINO
#tree <- rpart(Class ~ ., data = train, method = "class") ALREADY USED 
#predict.tree <- predict(tree, test, type="class") ALREADY USED
#confusionMatrix(as.factor(predict.tree), as.factor(test$Drug)) ALREADY USED
#printcp(tree)
tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"] # 0.07462687 ALREADY USED
plotcp(tree) # GRAFICO DEL CP VALUE ALREADY USED

ptree <- prune(tree, best)#cp=tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"], )
rfancyRpartPlot(ptree, uniform=TRUE, main="Pruned Classification Tree")

predict.ptree <- predict(ptree,test,type="class")
confusionMatrix(as.factor(predict.ptree), as.factor(test$Drug))


---
ptree <- prune(tree, cp= tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"], )
fancyRpartPlot(ptree, uniform=TRUE, main="Pruned Classification Tree")

predict.ptree <- predict(ptree,test,type="class")
confusionMatrix(as.factor(predict.ptree), as.factor(test$Drug))



--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
#----------------------------------------------------

#-------------------------------
# SVM
#n <- nrow(df_b)  # Number of observations
#ntrain <- round(n*0.75)  # 75% for training set
#set.seed(314)    # Set seed for reproducible results
#tindex <- sample(n, ntrain)   # Create a random index
#train_dfb <- df_b[tindex,]   # Create training set
#test_dfb <- df_b[-tindex,]   # Create test set
svm1 <- svm(Class~., data=train, 
            method="C-classification", kernal="radial", 
            gamma=0.1, cost=10)

summary(svm1)
prediction <-  predict(svm1, test_dfb)
xtab <- table(test_dfb$Class, prediction)
xtab

# PLOTTING SVM
plot(svm1, train, Popularity ~ energy, fill = TRUE, grid = 50, slice = list(key =0.864806, mode = 0.7763, speechiness = 0.49, acousticness = 0.74, instrumentalness = 0.5882, liveness = 0.5591, valence = 0.80, time_signature = 0.22, Loudness = 0.60, Tempo = 0.7321, Minutes = 3.72), symbolPalette = rainbow(4), color.palette = terrain.colors, svSymbol = "x", dataSymbol = "o")




slice = list(key =0.864806, mode = 0.7763, speechiness = 0.49, acousticness = 0.74, instrumentalness = 0.5882, liveness = 0.5591, valence = 0.80, time_signature = 0.22, Loudness = 0.60, Tempo = 0.7321, Minutes = 3.72)




symbolPalette = rainbow(4),
color.palette = terrain.colors
?palette
view(train)
# S3 method for svm
plot(x, data, formula, fill = TRUE, grid = 50, slice = list(),
     symbolPalette = palette("default"), svSymbol = "x", dataSymbol = "o", ...)

summary(train)

## a simple example
data(cats, package = "MASS")
m <- svm(Sex~., data = cats)
plot(m, cats)

## more than two variables: fix 2 dimensions
data(iris)
m2 <- svm(Species~., data = iris)
plot(m2, iris, Petal.Width ~ Petal.Length,
     slice = list(Sepal.Width = 3, Sepal.Length = 4))

## plot with custom symbols and colors
plot(m, cats, svSymbol = 1, dataSymbol = 2, symbolPalette = rainbow(4),
     color.palette = terrain.colors)




#--------------------------------
#ACCURACY 
errorcito <- (((758+500+505+784+795+712+265+806+600+529+177)/nrow(test_dfb)))
errorcito = as.character(errorcito)
paste("The accuracy is", errorcito, sep=' ')

-----------------------------------------------------
  
#CODICE DI FEDE
  
  # ---------------------------------------
# Split Zanotti

set.seed(123)
df_split <- initial_split(df_b, strata = NULL)

dfb_train <- training(df_split)
dfb_test <- testing(df_split)

set.seed(123)
df_folds <- vfold_cv(dfb_train)

# -----------------------------------
# Train Test Split

smp_size <- floor(0.70 * nrow(df_b))
set.seed(123)
train_ind <- sample(seq_len(nrow(df_b)), size = smp_size)

train <- df[train_ind, ]
test <- df[-train_ind, ]

# ----------------------------------------
# LDA 
df_b$Class <-(as.factor(df_b$Class))
model <- lda(Class ~ ., train)           # AS FACTOR
attributes(lda)
predict <- predict(model, test)
mean(predict$class==test$Class) #ACCURACY 0,43 ... GOOGLE AND EXPLAIN WHY
#define data to plot

lda_plot <- cbind(train, predict(model))
?cbind
#create plot
ggplot(lda_plot, aes(LD1, LD2)) +
  geom_point(aes(color = as.factor(Drug)))

# -----------------------------------------
# KNN
# KNN Naive
#KNN ONLY WORKS WITH AS.NUMERIC
df_b$Class <- as.numeric(as.factor(df_b$Class)) #already used 
train_labels <- train$Class
test_labels <- test$Class
nrow <- round(sqrt(nrow(train)))
train_labels
predictions <- knn(train, test, train_labels, k = nrow)
confusionMatrix(predictions,as.factor(test$Class))
# KNN Hp Tuning
head(train)
k.optm=1
i = 1
for (i in 1:100) {
  knn.mod <- knn(train, test, train_labels, k = i)
  k.optm[i] <- 100 * sum(test_labels == knn.mod)/NROW(test_labels)
  k=i
  cat(k, '=', k.optm[i], '\n')
}

# Best K is 1
predictions_optimized <- knn(train, test, train_labels, k = 28)
confusionMatrix(predictions_optimized,as.factor(test$Class))

# ------------------------------------------
# Tree Predictor

tree <- rpart(Drug ~ ., data = train, method = "class")
predict.tree <- predict(tree, test, type="class")

confusionMatrix(as.factor(predict.tree), as.factor(test$Drug))

printcp(tree)
tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"] # 0.07462687

plotcp(tree)

ptree <- prune(tree, cp= tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"], )
fancyRpartPlot(ptree, uniform=TRUE, main="Pruned Classification Tree")

predict.ptree <- predict(ptree,test,type="class")
confusionMatrix(as.factor(predict.ptree), as.factor(test$Drug))




























anova <- aov(Class ~ . -Class, data=df)
anova$coefficients
summary(anova)

# Here we see the P-values. We can tell that xij are significative. 
# This mean that they have explanatory power within a very simple linear model.
# If we were to do a linear regression (not appropiate), we would not consider the variables that have a p-value>0,05.
# We are going to perform PCA, we would 





#CLASSIFICATION WITH LOGISTIC MULTINOMIAL
library(tidyverse)
install.packages("caret")
install.packages("nnet")
library(nnet)
library(caret)
library(dplyr)

# Split the data into training and test set
set.seed(123)
training.samples <-  df$Class %>% 
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- df[training.samples, ]
test.data <- df[-training.samples, ]

# Fit the model
model <- nnet::multinom(Class ~., data = df)
# Summarize the model
output <- summary(model)
print(output)


# P-VALUES 
z <- output$coefficients/output$standard.errors
p <- (1 - pnorm(abs(z), 0, 1))*2 # we are using two-tailed z test
print(p)

# Make predictions
predicted.classes <- model %>% predict(test.data)
head(predicted.classes)
# Model accuracy
mean(predicted.classes == test.data$Class)


# CROSS VALIDATION
library(caret)

#specify the cross-validation method
ctrl <- trainControl(method = "cv", number = 5)

#fit a regression model and use k-fold CV to evaluate performance
model1 <- train(Class ~., data = df, method = "lm", family='binomial', trControl = ctrl)

#view summary of k-fold CV               
print(model1)

Linear Regression 

10 samples
2 predictor

No pre-processing
Resampling: Cross-Validated (5 fold) 
Summary of sample sizes: 8, 8, 8, 8, 8 
Resampling results:
  
  RMSE      Rsquared  MAE     
3.018979  1         2.882348

Tuning parameter 'intercept' was held constant at a value of TRUE

# We use the following code to find out how many songs of each category. 
library(plyr)
count(df, "Class")
# We see that it is clearly unbalanced. At the time of applying a model, if we were not to take
# this into account, we would have biased results. 

# BOXPLOT POPULARITY
boxplot(df$Popularity ~ df$Class)

#  DURATION IN MINUTES
#  vedere se ci sono differenze significativi per le medie


