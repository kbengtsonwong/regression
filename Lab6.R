setwd("/Users/kevinwong2014/documents/usf/fall2018/linearregression")
df <- read.csv(file = "hitters.csv", header = T)
n <- dim(df)[1]

df[1:5,]

set.seed(1)
train_idx <- sample(1:n, size = 210)
test_idx <- (1:n)[-train_idx]

length(train_idx)
length(test_idx)

library(glmnet)

# generating values for lambda in the range .001 to 10^10 
grid <- 10^seq(-3, 10, length = 1000)

X <- model.matrix(Salary ~ ., df)[,-1]
y <- df$Salary

# ridge regression
ridge.mod <- glmnet(X[train_idx,], y[train_idx], alpha=0, lambda=grid)
plot(ridge.mod, xvar = "lambda", label = TRUE)

# lasso regression
lasso.mod <- glmnet(X[train_idx,], y[train_idx], alpha=1, lambda=grid)
plot(lasso.mod, xvar = "lambda", label = TRUE)

# k-fold cross validation with default 10 folds to find the optimal lambda value for ridge 
cv.out <- cv.glmnet(X[train_idx,], y[train_idx], alpha=0)
plot(cv.out)
bestlam.r <- cv.out$lambda.min
bestlam.r
predict(ridge.mod, s=bestlam.r, type = "coefficients") 

# repeating the process for lasso
cv.out <- cv.glmnet(X[train_idx,], y[train_idx], alpha=1)
plot(cv.out)
bestlam.l <- cv.out$lambda.min
bestlam.l
predict(lasso.mod, s=bestlam.l, type = "coefficients")[1:12,]

# The comparison between the optimal lasso and ridge models begins with an examination of the lambda term.  
# Using lasso, the lambda is a small value close to 1 whereas the lambda associated with ridge is an order of magnitude larger at 34.  
# Moreover, the variable inclusion/exlclusion is different as lasso regularization allows for some beta terms to be shrunk completely to 0 and thus eliminated from the model.
# In contrast ridge regularization shrinks these same terms that were eliminated instead to values close to 0.  
# The last difference between the two is that the beta values for terms in common can differ not insignificantly.

# Calculating predictive RMSE for ridge regression
ridge.pred <- predict(ridge.mod, s=bestlam.r, newx=X[test_idx,])
rmse_ridge <- sqrt(mean((ridge.pred - y[test_idx])^2))

# Calculating predictive RMSE for lasso regression
lasso.pred <- predict(lasso.mod, s=bestlam.l, newx=X[test_idx,])
rmse_lasso <- sqrt(mean((lasso.pred - y[test_idx])^2))

# Stepwise Model found from Lab 5
library(boot)
glm_stepwise <- glm(Salary ~ CRBI + Hits + PutOuts + Division + AtBat +  Walks + CWalks + CRuns + CAtBat + Assists, data = df)
stepwise.pred <- predict(object = glm_stepwise, newdata = df[test_idx,])
rmse_stepwise <- sqrt(mean((stepwise.pred - y[test_idx])^2))

results <- data.frame(r = ridge.pred, l = lasso.pred, stepwise = stepwise.pred, actual = y[sort(test_idx)])
colnames(results) <- c("ridge","lasso","stepwise","actual")
results
