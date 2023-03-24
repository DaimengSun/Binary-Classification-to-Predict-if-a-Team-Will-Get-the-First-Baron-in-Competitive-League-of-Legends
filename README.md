# Binary-Classification-to-Predict-if-a-Team-Will-Get-the-First-Baron-in-Competitive-League-of-Legends

by Daimeng Sun (dsun@ucsd.edu)<br>
Our exploratory data analysis on this dataset can be found <a href="https://daimengsun.github.io/League-of-Legends-Match-Data-Analysis/">here</a>


---
### Framing the Problem

A binary classifier is built based on the real-world League of Legends dataset, to predict if a team will get the first baron in the game. <br><br> The response variable is a boolean indicating if a team will get the first baron in the game. <br><br> Accuracy is used to evaluate the model because correct predictions and incorrect predictions are equally detrimental. Thus, the model is evaluated upon both true and false positives and negatives. $$\text{accuracy} = \frac{TP + TN}{TP+FP+FN+TN}$$<br><br> Features that contain information post baron spawn (20 minutes into game), such as total gold difference, dragon kills, tower taken, are excluded from potential features becasue they are information we wouldn't know at the time of prediction. Features that contain overall game information, such as game result and game length, are also excluded because of same reason. <br><br> To reduce multicollinearity, features that exhibit strong correlation are also excluded from potential features. For example, cs difference at 15 and gold difference at 10 are excluded because gold difference at 15 is selected as one of the potential features.

Dataframe after cleaning:<br>
|   patch | side   | firstdragon   |   golddiffat15 |   xpdiffat15 |   heralds |   firstbaron |
|--------:|:-------|:--------------|---------------:|-------------:|----------:|-------------:|
|   12.01 | Blue   | False         |            107 |        -1617 |         2 |            0 |
|   12.01 | Red    | True          |           -107 |         1617 |         0 |            0 |
|   12.01 | Blue   | False         |          -1763 |         -906 |         1 |            0 |
|   12.01 | Red    | True          |           1763 |          906 |         1 |            1 |
|   12.01 | Blue   | True          |           1191 |         2298 |         1 |            1 |

---
### Baseline Model

The baseline binary classifier is built using class DecisionTreeClassifier from sklearn.tree, with a max-depth of two. During a League of Legends competitive game, the classifier predicts whether a team will get the first baron (spawns at 20 minutes), based on golddiffat15 (team's gold difference at 15 minutes), and heralds (number of heralds the team has claimed). <br><br>
"golddiffat15" is calculated using formula: golddiffat15 = team total gold - opponent total gold <br>It is a quantitative data and will not undergo any extra encoding methods. This feature is selected because in a League of Legends competitive game, team with an early lead is likely to proceed to claim first baron and win.<br><br>
"heralds" is also a quantitative data but there are only three possible values: 0, 1, and 2. A binarizer with threshold of 1 is used to encode data for this feature. This is because in a League of Legends competitive game, the first herald is much more important than the second herald. Leading team may choose to claim dragon and leave the second herald to opponent. Only teams with dominating advantage are likely to claim both heralds, then proceed to claim first baron.<br><br>

Baseline model performance analysis:<br><br>
The baseline model has an accuracy of 0.685 on the testing data. It is difficult to judge if an accuracy of 0.685 indicates good model performance or bad. On one hand, an accuracy of 0.685 is not optimal for a binary classifier and the a tree depth of two may not be the optimal tree depth. A search for the best hyperparameters could be done to improve the model. On the otherhand, there is no guarantee that the leading team will claim the first baron, because League of Legends games are so dynamic. The other team could win one teamfight and claim the first baron, or they can steal the first baron by completing the finishing hit on first baron.<br>
Visualization of base model
<iframe src="assets/basetree.html" width=800 height=600 frameBorder=0></iframe>

---
### Final Model

In contrast to the baseline classifier, three more features is added in attempt to better predict if a team will claim the first baron during an League of Legends competitive game. These three features are: side (blue or red, categorical), xpdiffat15 (total experience difference at 15 minutes, quantitative), and firstdragon (if the team claims first dragon, boolean).<br><br>
"side" is a categorical data that indicates which side of the map the team is playing on. As discovered in the EDA, due to asymmetry, there might be innate advantages for the team on the blue side. This feature is OneHotEncoded and included to address this potential factor.<br><br>
"firstdragon" is a boolean that indicates whether the team claims the first dragon in the game. As discussed above regarding baseline model, leading teams in competitive League of Legends might not be able to claim all the early objectives. This feature is OneHotEncoded and added to the model to hopefully better predict teams with a huge early lead and proceed to claim the first baron.<br><br>
"xpdiffat15" is a quantitative data similar to "golddiffat15", beside it is comparing total experience difference at 15 minutes. Experience reflects a team's overall levels, which has a significant impact during combats but is often times overlooked. This features is added to hopefully better predict team's combats ability beside gold difference.<br><br>
In addition to new features, to improve the classifier's performance (accuracy), GridSearchCV from sklearn.model_selection is used to decide the best-performing hyperparameters for the decision tree. Hyperparameter tested include 'max_depth': [2, 3, 4, 5, 6, 8, 10, 15, 18, None], 'min_samples_split': [2, 5, 10, 20, 50, 100, 200], and 'criterion': ['gini', 'entropy']. Each combination of hyperparameters are fitted using training data and their performance is tested using cross-validation with k-fold of five..<br><br>
After finding out the best performing hyperparameters, the model is then fitted using training data, with features as ['firstdragon', 'side','xpdiffat15', 'golddiffat15', 'heralds'], and with hyperparameters as {'criterion': 'gini', 'max_depth': 4, 'min_samples_split': 2}. "firstdragon" and "side" are one hot encoded, "heralds" is binarized, "xpdiffat15" and "golddiffat15" are used as is.<br>
Confusion matrix of final model:
<iframe src="assets/confusion.html" width=800 height=600 frameBorder=0></iframe>

Improvement investigation: The final model has an accuracy of 0.691 on the testing data, which achieved a 0.006 improvement on accuracy.

---
### Fairness Analysis

Null Hypothesis: The classifier's accuracy is the same for both blue side and red side, and any differences are due to chance.<br>
Alternative Hypothesis: The classifier's accuracy is higher for red side.<br>
Test statistic: Difference in accuracy (blue minus red).<br>
Significance level: 0.01.<br>
<iframe src="assets/Observed Difference in Accuracy.html" width=800 height=600 frameBorder=0></iframe>
Conclusion: We fail to reject the null hypothesis and it seems like the difference in accuracy across two sides is due to random chance.

---