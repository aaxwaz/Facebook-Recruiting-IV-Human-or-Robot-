setwd('c:\\dataset')
rm(list = ls())
require(xgboost)
require(randomForest)
require(e1071)
require(TunePareto)
require(ROCR)
require(adabag)

# kfold stratified
CV_split <- function(train_label, K){
  result = generateCVRuns(train_label, ntimes = 1, nfold = K, stratified = T)
  result[[1]]
}

evaluation = function(probs, target, eps = 1e-15){
  probs[probs < eps] = eps
  probs[probs > 1- eps] = eps
  pred = prediction(probs, target)
  perf = performance(pred, 'auc')
  attributes(perf)['y.values'][[1]][[1]]
}

df = read.csv('train.csv')
df$bider_id = NULL
df$outcome = as.factor(df$outcome)

TOTAL_RUNS = 5 # total rounds of CV 
K = 10 # K-fold stratified CV
bagSize = 20 # bagging size  

## model params 
boos = TRUE # adaboost
mfinal = 10
coeflearn = 'Breiman'
ntree = 60 # rf
mtry = 9
gamma = 0.01 # svm
cost = 3

##### CV process #####
total_scores = rep(0,TOTAL_RUNS)
for(r in 1:TOTAL_RUNS){
  CV_index = CV_split(df$outcome, K)
  auc_scores = rep(0,K)
  cat("Scores: ")
  for (cv in 1:K)
  {
    #cat("CV: ", cv, "\n")
    valid_index = CV_index[[cv]]
    
    template = 1:nrow(df)
    valid_template = template %in% valid_index
    
    valid_x = df[valid_template,]  # validating data set
    train_x = df[!valid_template,]  # training data set
    
    # not scaled, for rf 
    train_x_rf = train_x
    valid_x_rf = valid_x
    
    # log transform
    train_x[,1:(ncol(train_x)-1)] = log(1.0 + train_x[,1:(ncol(train_x)-1)])
    valid_x[,1:(ncol(valid_x)-1)] = log(1.0 + valid_x[,1:(ncol(valid_x)-1)])
    
    trind = 1:nrow(train_x)
    #teind = 1:nrow(valid_df) no use 
    
    ### start bagging ###
    baggingRuns = 1:bagSize
    pred_final = 0
    for (z in baggingRuns) {
      #cat(z, ' ')
      #bag_index = sample(trind,size=as.integer(length(trind)),replace=T)
      #OOB_index = setdiff(trind,bag_index)
      
      ### stratified split bag and OOB
      stra_index = CV_split(train_x[,ncol(train_x)], 4) # 75%
      OOB_index_temp = stra_index[[1]]
      OOB_index = trind %in% OOB_index_temp
      bag_index = setdiff(trind,OOB_index)
      
      X_rf = train_x_rf[OOB_index,-ncol(train_x_rf)]
      y_rf = train_x_rf[OOB_index, ncol(train_x_rf)]
      
      X_svm = train_x[OOB_index,-ncol(train_x)]
      y_svm = train_x[OOB_index, ncol(train_x)]
      
      # train models on OOB sets
      rf_model = randomForest(x=X_rf, y=as.factor(y_rf), replace=T, ntree=ntree, do.trace=F, mtry=mtry)
      svm_model = svm(as.factor(y_svm)~., data = X_svm, gamma = gamma, cost = cost, class.weights=c('0'=0.1,'1'=1.0), 
                      probability = TRUE)
      
      # assign bagging sets to ADA
      X_ada = train_x[bag_index,-ncol(train_x)]
      y_ada = train_x[bag_index, ncol(train_x)]
      
      # predict rf and svm on bagging sets
      rf_pred = predict(rf_model, X_ada, type="prob")
      rf_pred = rf_pred[,2]
      svm_pred = predict(svm_model, X_ada, probability=TRUE)
      svm_pred = attr(svm_pred,'probabilities')[,2]
      
      # predict on valid sets 
      rf_pred_valid = predict(rf_model, valid_x_rf[,-ncol(valid_x_rf)], type="prob")
      rf_pred_valid = rf_pred_valid[,2]
      svm_pred_valid = predict(svm_model, valid_x[,-ncol(valid_x)], probability=TRUE)
      svm_pred_valid = attr(svm_pred_valid,'probabilities')[,2]
      
      ## train xgboost model with probabilities from previous preds 
      # combine new training and validating dfs
      train_ada = cbind(X_ada, temp=(rf_pred*svm_pred)^0.5, outcome =y_ada)
      
      valid_ada_X = cbind(valid_x[,-ncol(valid_x)], temp=(rf_pred_valid*svm_pred_valid)^0.5, outcome=valid_x[,ncol(valid_x)])
      
      # train ada
      ada_model = boosting(outcome~., data = train_ada, boos = boos, mfinal = mfinal, coeflearn = coeflearn)
      
      # predict 
      pred = predict(ada_model, valid_ada_X, probability = TRUE)
      pred = pred$prob[,2]
      
      pred_final = pred_final + pred
      #pred_final = pred_final + rank(pred, ties.method = "random")
      
    } # end of bagging 
    #pred_final = seq(from=0, to=1, length.out=nrow(valid_x))[rank(pred_final, ties.method = "random")]
    pred_final = pred_final / z
    auc_scores[cv] = evaluation(pred_final, valid_x[,ncol(valid_x)])
    cat(auc_scores[cv], " ")
  } # end of CV
  cat("\n")
  print(auc_scores)
  cat("mean: ", mean(auc_scores), "sd: ", sd(auc_scores))
  total_scores[r] = mean(auc_scores)
}

print("\nTOTAL RUns SCORES: \n")
print(total_scores)
print(mean(total_scores))

##### SUBMISION! #####
## data loading and preparation
df = read.csv('train.csv')
df$bider_id = NULL
submit_df = read.csv('test.csv')
test_id = as.character(submit_df$bider_id)
submit_df$bider_id = NULL
x = as.matrix(df)
x = matrix(as.numeric(x),nrow(x),ncol(x))
submit_x = as.matrix(submit_df)
submit_x = matrix(as.numeric(submit_x),nrow(submit_x),ncol(submit_x))

x_rf = x # rf do not scale
x[,1:(ncol(x)-1)] = log(1.0 + x[,1:(ncol(x)-1)])
submit_x_rf = submit_x
submit_x = log(1.0 + submit_x)
# dummy col 
submit_x$outcome = 1
submit_x$outcome[1] = 0
submit_x$outcome = as.factor(submit_x$outcome)


## model params 
ntree = 60 # rf
mtry = 9
gamma = 0.005 # svm
cost = 2
boos = TRUE # ada
mfinal = 10
coeflearn = 'Breiman'

### start bagging 
trind = 1:nrow(x)

baggingSize = 1:80
pred_final = 0
for (z in baggingSize) {
  cat(z, " \n")
  
  stra_index = CV_split(x[,ncol(x)], 4) # 75%
  OOB_index_temp = stra_index[[1]]
  OOB_index = trind %in% OOB_index_temp
  bag_index = setdiff(trind,OOB_index)
  
  X_rf = x_rf[OOB_index,-ncol(x_rf)]
  y_rf = x_rf[OOB_index, ncol(x_rf)]
  
  X_svm = x[OOB_index,-ncol(x)]
  y_svm = x[OOB_index, ncol(x)]
  
  # train models on OOB sets
  rf_model = randomForest(x=X_rf, y=as.factor(y_rf), replace=T, ntree=ntree, do.trace=F, mtry=mtry)
  svm_model = svm(as.factor(y_svm)~., data = X_svm, gamma = gamma, cost = cost, class.weights=c('0'=0.1,'1'=1.0),
                  probability = TRUE)
  
  # assign bagging sets to XGB
  X_ada = x[bag_index,-ncol(x)]
  y_ada = x[bag_index, ncol(x)]
  
  # predict rf and svm on bagging sets
  rf_pred = predict(rf_model, X_ada, type="prob")
  rf_pred = rf_pred[,2]
  svm_pred = predict(svm_model, X_ada, probability=TRUE)
  svm_pred = attr(svm_pred,'probabilities')[,2]
  
  # predict on submit sets 
  rf_pred_submit = predict(rf_model, submit_x_rf, type="prob")
  rf_pred_submit = rf_pred_submit[,2]
  svm_pred_submit = predict(svm_model, submit_x[,-ncol(submit_x)], probability=TRUE)
  svm_pred_submit = attr(svm_pred_submit,'probabilities')[,2]
  
  ## train ada model with probabilities from previous preds 
  train_ada = cbind(X_ada, temp=(rf_pred*svm_pred)^0.5, outcome =y_ada)
  
  submit_ada_X = cbind(submit_x[,-ncol(submit_x)], temp=(rf_pred_submit*svm_pred_submit)^0.5, outcome=submit_x[,ncol(submit_x)])
  
  # train ada and predict on submit_ada_x
  ada_model = boosting(outcome~., data = train_ada, boos = boos, mfinal = mfinal, coeflearn = coeflearn)
  pred = predict(ada_model, submit_ada_X, probability = TRUE)#, probability = TRUE)
  pred = pred$prob[,2]
  
  pred_final = pred_final + pred
  
} # end of bagging 
pred_final = pred_final / z

### prepare submission 
submitFile = read.csv('sampleSubmission.csv')
submitFile$bidder_id = as.character(submitFile$bidder_id)
for (i in 1:4630){
  submitFile$prediction[submitFile$bidder_id == test_id[i]] = pred_final[i]
}
write.csv(submitFile, 'bag_adaboost_pred.csv', row.names = FALSE)  






