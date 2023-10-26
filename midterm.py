
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb

df = pd.read_csv("Employee.csv")


df.describe()


df.columns = df.columns.str.lower()


# Now let's change how we measure the years in the company by changing the joining year by the years in the company.


df["yearsinthecompany"] = 2024 - df["joiningyear"]

payemnet_tier_map = {
    1: "first_tier",
    2: "second_tier",
    3: "third_tier"
}
df["paymenttier"] = df["paymenttier"].map(payemnet_tier_map)

categorical_columns = df.dtypes[df.dtypes == object].index
for column in categorical_columns:
    df[column] = df[column].str.lower().str.replace(" ", "_")


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=1)
print(len(df_train), len(df_test), len(df_val))


df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


y_train = df_train["leaveornot"]
y_val = df_val["leaveornot"]
y_test = df_test["leaveornot"]
y_full_train = df_full_train["leaveornot"]

del df_train["leaveornot"]
del df_val["leaveornot"]
del df_test["leaveornot"]
del df_full_train["leaveornot"]


dv = DictVectorizer(sparse=False)
encoded_df_train = dv.fit_transform(df_train.to_dict(orient="records"))
encoded_df_val = dv.transform(df_val.to_dict(orient="records"))


dt = DecisionTreeClassifier(max_depth=7, min_samples_leaf=3)
dt.fit(encoded_df_train, y_train)
y_predict = dt.predict(encoded_df_val)


roc_auc = roc_auc_score(y_val, y_predict)


min_sample_leafes = range(1, 20)
max_depths = range(1, 20)
scores = []
for mleafs in min_sample_leafes:
    for depth in max_depths:
        dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=mleafs)
        dt.fit(encoded_df_train, y_train)
        y_predict = dt.predict(encoded_df_val)
        roc_auc = roc_auc_score(y_val, y_predict)
        scores.append((mleafs, depth, roc_auc))

scores_df = pd.DataFrame(
    scores, columns=["min_sample_leafes", "max_depth", "roc_auc_score"])


scores_df.sort_values(by="roc_auc_score", ascending=False)


#  The best parameters are 7 for max_depth and min_leaf_samples 1


dt = DecisionTreeClassifier(max_depth=7, min_samples_leaf=1)
dt.fit(encoded_df_train, y_train)
y_predict = dt.predict(encoded_df_val)
roc_auc = roc_auc_score(y_val, y_predict)


rf_scores = []
for n in range(10, 201, 10):
    for depth in range(1, 16):
        rf = RandomForestClassifier(
            n_estimators=n, max_depth=depth, random_state=1)
        rf.fit(encoded_df_train, y_train)
        y_predict = rf.predict(encoded_df_val)
        roc_auc = roc_auc_score(y_val, y_predict)
        rf_scores.append((n, depth, roc_auc))
rf_scores_df = pd.DataFrame(
    rf_scores, columns=["n_estimators", "max_depth", "roc_auc_score"])
rf_scores_df.sort_values(by="roc_auc_score", ascending=False)


# So for the best model in random forest we use n_estimators 180 , max_depth = 8


rf = RandomForestClassifier(n_estimators=180, max_depth=8, random_state=1)
rf.fit(encoded_df_train, y_train)
y_predict = rf.predict(encoded_df_val)
rf_roc_auc = roc_auc_score(y_val, y_predict)


# Now let's try xgboost


feature_names = list(dv.get_feature_names_out())
d_train_xgb = xgb.DMatrix(
    encoded_df_train, label=y_train, feature_names=feature_names)
d_val_xgb = xgb.DMatrix(encoded_df_val, label=y_val,
                        feature_names=feature_names)


xgb_scores = []
for eta in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    for depth in range(1, 16):
        x_params = {
            "eta": eta,
            "max_depth": depth,
            "min_child_weight": 1,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "seed": 1,
            "verbosity": 1
        }
        xgb_model = xgb.train(x_params, d_train_xgb, num_boost_round=20)
        y_pred_xgb = xgb_model.predict(d_val_xgb)
        roc_xgb = roc_auc_score(y_val, y_pred_xgb)
        xgb_scores.append((eta, depth, roc_xgb))
xgb_scores = pd.DataFrame(
    xgb_scores, columns=["eta", "max_depth", "xgb_scores"])
xgb_scores.sort_values(by="xgb_scores", ascending=False)


# so the best xgb model is max_depth 7 and eta 0.3


x_params = {
    "eta": 0.3,
    "max_depth": 7,
    "min_child_weight": 1,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "seed": 1,
    "verbosity": 1
}
chosen_xgb_model = xgb.train(x_params, d_train_xgb, num_boost_round=20)
y_pred_xgb = chosen_xgb_model.predict(d_val_xgb)
roc_xgb = roc_auc_score(y_val, y_pred_xgb)


# Comparing this model to the previous models of Decision Tree roc => 0.828  \
#   and Random forest of roc => 0.829
#   it has a better roc score.


# Now let's train it on the full train data set


full_dv = DictVectorizer(sparse=False)
encoded_df_full_train = full_dv.fit_transform(
    df_full_train.to_dict(orient="records"))
encoded_df_test = full_dv.transform(df_test.to_dict(orient="records"))


full_feature_names = list(full_dv.get_feature_names_out())
d_full_train_xgb = xgb.DMatrix(
    encoded_df_full_train, label=y_full_train, feature_names=full_feature_names)
d_test_xgb = xgb.DMatrix(encoded_df_test, label=y_test,
                         feature_names=full_feature_names)


x_params = {
    "eta": 0.3,
    "max_depth": 7,
    "min_child_weight": 1,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "seed": 1,
    "verbosity": 1
}
chosen_xgb_model = xgb.train(x_params, d_full_train_xgb, num_boost_round=20)
y_pred_xgb = chosen_xgb_model.predict(d_test_xgb)
roc_xgb = roc_auc_score(y_test, y_pred_xgb)
roc_xgb


# The Total ROC score after training the model over the full data set has decreased a little but still has an accepted ROC


# Now let's export the model using pickle


pickle.dump(chosen_xgb_model, open("emloyeeleaveclassifier.pkl", "wb"))
pickle.dump(full_dv,open("dictvictorizer.pkl","wb"))