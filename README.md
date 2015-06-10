# Facebook-Recruiting-IV-Human-or-Robot-
Final solution for Facebook Recruiting IV competition on kaggle

This is my solution for the Kaggle competition - Facebook Recruiting IV: Human or Robot?
https://www.kaggle.com/c/facebook-recruiting-iv-human-or-bot/leaderboard/private

With the 2-stage adaboost model attached in the repo, you could get a private LB score of 0.936, which is around position of 22 out of 985 teams. 

However, before the model can be trained, you must apply feature engineerings in order to extract and compile new data set out of the raw data provided by the competition. The features I have created can be summarised below: 

1. Total bids for each bidder

2. Total auctions for each bidder

3. Total bids in each country for each bidder

4. Total urls for each bidder

5. Total unique times for each bidder

6. Total devices for each bidder

7. Max number of bids that share the same left-most-5 digits of time for each bidder

8. Max number of bids that share the same left-most-4 digits of time for each bidder

9. Max number of bids that share the same left-most-6 digits of time for each bidder

10. The median time difference of all consecuitive bids for each bidder

11. Total ips for each bidder

12. Total ips in Indian for each bidder

13. Total bids of each merchandise for each bidder

14. The median of each auction's time for each bidder.

The final model attached is a 2-stage adaboost (bagging & stacking). In stage 1, we train randomForest and SVM (base model) on OOB data, and in Stage 2, the adaboost (meta model) is trained on BAG data with predictions from stage 1 combined. And we repeat the iterations 80 times as bagging process. 

The BAG data is 75% of the original training data randomly sampled WITHOUT replacement. The OOB is the remaining 25%. 

The final prediction is the average of all the predictions from each iteration. 
