# Imports
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy import stats
from scipy.stats import pointbiserialr, chi2, chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, precision_recall_curve, \
    average_precision_score, auc, f1_score, accuracy_score, recall_score, precision_score, \
    confusion_matrix, make_scorer

# ------------------------------------------------------------------------
# Data Cleaning and pre-processing
# 1. Check for missing data
# 2. CHeck for imbalanced data
# 3. Format modification
# 4. Preliminary feature engineering
# 5. Cap and floor certain variables
# ------------------------------------------------------------------------
pd.set_option('display.float', '{:.2f}'.format)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 200)

# Read data
df = pd.read_csv("UCI_Credit_Card.csv")
print(df.info())
print(df.head())
print(df.describe())

# 1. Check for missing data ---------------------------------------¬
pct_missing = df.isnull().mean().round(4).mul(100).sort_values(ascending=False)  # no missing
print(pct_missing)

# 2. Check for imbalanced data

plt.figure(figsize=(12, 10))
plt.title('Default Credit Card \n (Default = 1, Not Default = 0')
cc_default = df.groupby(['default.payment.next.month'])['default.payment.next.month'].count()
cc_default_plot = cc_default.plot(colormap='Set3', kind='bar')
for p in cc_default_plot.patches:
    width = p.get_width()
    height = p.get_height()
    x, y = p.get_xy()
    cc_default_plot.annotate(f'{height}', (x + width / 2, y + height * 1.02), ha='center')
plt.savefig('Default_Count_plot.png', bbox_inches='tight')
plt.close()

# 3. Modify format
df.rename(columns={'default.payment.next.month': 'def'}, inplace=True)
# change payment indicators, sex, education, marriage status to categorical variable
df['SEX'] = df['SEX'].apply(lambda x: "Male" if (x == 1) else "Female")
df['MARRIAGE'] = df['MARRIAGE'].apply(lambda x: "married" if (x == 1) else ("single" if (x == 2) else "other"))
df['EDUCATION'] = df['EDUCATION'].apply(lambda x: "Grad" if (x == 1) else (
    "Uni" if (x == 2) else ("Hi-sch" if (x == 3) else ("other" if (x == 4) else "unknown"))))
# Drop ID
df.drop(['ID'], axis=1, inplace=True)  # don't need ID

# PAY variable has -2 and 0 values that are not documented, assume that they equal to pay duly
for i in (0, 2, 3, 4, 5, 6):
    pay = "PAY_" + str(i)
    df[pay] = df[pay].apply(lambda x: -1 if (x == -2 or 0) else x)

# 4. Preliminary feature engineering
# compute utilization rate, which is a good indicator of the borrower's account status
for i in range(1, 7):
    bill = 'BILL_AMT' + str(i)
    util = 'UTIL' + str(i)
    df[util] = df[bill] * 100 / df['LIMIT_BAL']


# compute utilization rate increase indicator - continuously increasing util rate signals financial stress
def conditions(s):
    if df[util_t1] > df[util_t2]:
        return 1
    else:
        return 0


for i in range(1, 6):
    util_t0 = 'UTIL' + str(i - 1)
    util_t1 = 'UTIL' + str(i)
    util_t2 = 'UTIL' + str(i + 1)
    if i == 1:
        df.loc[df[util_t1] > df[util_t2], 'mths_inc_' + str(i)] = i
    else:
        df.loc[(df[util_t0] > df[util_t1]) & (df[util_t1] > df[util_t2]), 'mths_inc_' + str(i)] = df['mths_inc_' + str(
            i - 1)] + 1
    # Calculate consecutive mths
    df.loc[df['mths_inc_' + str(i)] == i, 'cons_util_inc'] = i

df['cons_util_inc'].fillna(0, inplace=True)
df = df[df.columns.drop(list(df.filter(regex='mths_inc_')))]
print(df)

# 5. Cap and floor certain variables
# Floor negative bill balances (bill credits) at 0
# Floor negative utilization rates at 0
for i in range(1, 7):
    bill = 'BILL_AMT' + str(i)
    util = 'UTIL' + str(i)
    df[util] = df[util].apply(lambda x: 0 if (x < 0) else x)
    df[bill] = df[bill].apply(lambda x: 0 if (x < 0) else x)

print(df.describe())

# ------------------------------------------------------------------------------------
# Data exploration
# 1. Preliminary Screening - Uni-variate and Bi-variate Analysis
#     a. Compute Point Bi-serial Correlation for Categorical Y vs Continuous X
#     b. Compute Chi-squared statistics for Categorical Y vs Categorical
# 2. Filter Multicollinearity
# 3. Outlier filter
# ------------------------------------------------------------------------------------
# Divide into 1.Categorical 2.Numerical 3.Dependent Variables
df_X = df.drop(['def'], axis=1)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
categorical = ['object', 'category']
df_num = df_X.select_dtypes(include=numerics).copy()
df_cat = df_X.select_dtypes(include=categorical).copy()
df_def = df[['def']]
print('Numeric', df_num.shape)
print('Categorical', df_cat.shape)


# Preliminary Screening:
# 1. Compute Point Bi-serial Correlation for Categorical Y vs Continuous X
# 2. Compute Chi-squared statistics for Categorical Y vs Categorical Y

def prescreen(datain, defdata, numdata, catdata, isnum=True):
    if isnum:
        pbc_corr = []
        pbc_pvalue = []
        pbc_num = []

        for num in numdata.columns:
            pbc = pointbiserialr(defdata['def'], numdata[num])
            pbc_corr.append(pbc[0])
            pbc_pvalue.append(pbc[1])
            pbc_num.append(num)
            pbc_df = pd.DataFrame({'Numerical Var': pbc_num, 'PBC Corr': pbc_corr, 'P-value': pbc_pvalue})
            sns.displot(data=datain, x=num, kind='hist', height=6, aspect=2.5, bins=20, hue='def',
                        palette="PRGn")
            plt.savefig(num + 'dist_plot.png', bbox_inches='tight')
            plt.close()
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))
            sns.boxplot(ax=ax1, x="def", y=num, hue="def", data=datain, palette="PRGn", showfliers=True)
            sns.boxplot(ax=ax2, x="def", y=num, hue="def", data=datain, palette="PRGn", showfliers=False)
            plt.savefig(num + 'box_plot.png', bbox_inches='tight')
            plt.close()

        # Selection Criteria: abs(PBC Corr) > 0.05 and p-value < 0.05
        pbc_keep = pbc_df.loc[((abs(pbc_df['PBC Corr']) > 0.05) & (pbc_df['P-value'] < 0.05))]
        pbc_df.to_csv(r'Point Biserial Correlation Test Results.csv', index=None)
        print(pbc_keep.sort_values(by=['PBC Corr'], key=abs, ascending=False))

        # Only retain numerical variables that have significant correlation with target event
        pbc_filter = pbc_df.loc[
            ((abs(pbc_df['PBC Corr']) <= 0.05) | (pbc_df['P-value'] >= 0.05)), 'Numerical Var']
        print('Filtered variables out based on PBC \n', pbc_filter)
        df.drop(pbc_filter, axis=1, inplace=True)
        df_num.drop(pbc_filter, axis=1, inplace=True)
        print('Shape after filter out based of PBC \n', df.shape)

    elif not isnum:
        original_stdout = sys.stdout
        chi_critical = []
        chi_stat = []
        chi_pvalue = []
        chi_cat = []

        with open('Chi-Squared Test Results.txt', 'w') as f:
            for cat in catdata.columns:
                ax = (datain.groupby([cat, 'def'])['def'].count() / datain.groupby([cat])[
                    'def'].count()).unstack().plot(
                    kind='bar',
                    figsize=(
                        12, 8),
                    width=0.5,
                    colormap="Pastel2",
                    edgecolor=None)
                for p in ax.patches:
                    width = p.get_width()
                    height = p.get_height()
                    x, y = p.get_xy()
                    ax.annotate(f'{height:.0%}', (x + width / 2, y + height * 1.02), ha='center')

                plt.savefig(cat + '_pct_plot.png', bbox_inches='tight')
                plt.close()
                # Chi- Squared contingency table
                data_crosstab = pd.crosstab(df['def'], df_cat[cat], margins=False)
                print('\n', '-------------', cat, '-------------', '\n', data_crosstab)

                stat, p, dof, expected = chi2_contingency(data_crosstab)
                print('dof=%d' % dof)
                prob = 0.95
                critical = chi2.ppf(prob, dof)

                print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
                if abs(stat) >= critical:
                    print('Dependent (reject H0)')
                else:
                    print('Independent (fail to reject H0)')
                # Interpret p-value
                alpha = 1.0 - prob
                print('significance = %.3f, p=%.3f' % (alpha, p))
                if p <= alpha:
                    print('Dependent (reject H0)')
                else:
                    print('Independent (fail to reject H0)')
                chi_critical.append(critical)
                chi_stat.append(stat)
                chi_pvalue.append(p)
                chi_cat.append(cat)
                chi_df = pd.DataFrame(
                    {'Categorical Var': chi_cat, 'Chi-Sq Stat': chi_stat, 'Chi-Sq Critical': chi_critical,
                     'P-value': chi_pvalue})
                chi_df['alpha'] = alpha

            sys.stdout = original_stdout

            # Selection Criteria: Chi-Sq test stat > critical value and p-value < 0.05
            chi_keep = chi_df.loc[
                ((chi_df['Chi-Sq Stat'] >= chi_df['Chi-Sq Critical']) & (chi_df['P-value'] <= chi_df['alpha']))]
            chi_df.to_csv(r'Chi-Squared Test Results.csv', index=None)

            # Only retain numerical variables that have significant correlation with target event
            chi_filter = chi_df.loc[
                ((chi_df['Chi-Sq Stat'] < chi_df['Chi-Sq Critical']) & (
                        chi_df['P-value'] > chi_df['alpha'])), 'Categorical Var']
            print('Filtered variables out based on chi-sq \n', chi_filter)
            df.drop(chi_filter, axis=1, inplace=True)
            df_cat.drop(chi_filter, axis=1, inplace=True)
            print('Shape after filter out based of Chi-Sq \n', df.shape)


prescreen(df, df_def, df_num, df_cat, isnum=True)
prescreen(df, df_def, df_num, df_cat, isnum=False)

# Check for multicollinearity
X = add_constant(df_num)
vif = pd.Series([variance_inflation_factor(X.values, i)
                 for i in range(X.shape[1])],
                index=X.columns)
print(vif)
with open("VIF.txt", "w") as text_file:
    text_file.write("VIF Results: %s" % vif)

plt.figure(figsize=(35, 25))
corr = sns.heatmap(df_num.corr(), annot=True)
plt.savefig('Corr_plot.png', bbox_inches='tight')
plt.close()

df.drop(['PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'], axis=1, inplace=True)
df_num.drop(['PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'], axis=1, inplace=True)
print('Shape after filter out high mulitcollinearity \n', df.shape)

plt.figure(figsize=(35, 25))
corr = sns.heatmap(df_num.corr(), annot=True)
plt.savefig('Corr1_plot.png', bbox_inches='tight')
plt.close()

# Detect and drop outliers using 3 std threshold
print(' Before removing outliers', df.shape)
df = df[(np.abs(stats.zscore(df.select_dtypes(include=[np.number])) < 3).all(axis=1))]
print(' after removing outliers', df.shape)

# ------------------------------------------------------------------------------------
# More data pre-processing: Create dummy variables for categorical data
# ------------------------------------------------------------------------------------
# Create dummy variables and drop first for categorical data to avoid perfect multicollinearity
df_final = pd.get_dummies(df, columns=df_cat.columns, drop_first=False)

# --------------------------------------------------------------------------------------
#  Train, Test Split:
# --------------------------------------------------------------------------------------
X = df_final.drop(['def'], axis=1)
y = df_final[['def']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Check percentage of default event
print(y_train['def'].sum() / len(y_train))
print(y_test['def'].sum() / len(y_test))

# print(y[y.isnull().any(axis=1)])
# print(X[X.isnull().any(axis=1)])
# ----------------------------------------------------------------------------------------------
# Model Selection
# 1. XGBoost
# 2. Random Forest
# -----------------------------------------------------------------------------------------------
# --------------------------------- 1. XGBoost --------------------------------------
# Model parameter selection:
scoring = 'roc_auc'

# SMOTE
# 1. imbalance data, 22.7% default rate

stratified_kfold = StratifiedKFold(n_splits=3,
                                   shuffle=True,
                                   random_state=1)

pipeline_xgb = Pipeline(steps=[['smote', SMOTE(random_state=1)],
                               ['scaler', StandardScaler()],
                               ['classifier', XGBClassifier(objective='binary:logistic',
                                                            use_label_encoder=False,
                                                            missing=1,
                                                            seed=1,
                                                            subsample=0.8,
                                                            colsample_bytree=0.5)]])

# ['smote', SMOTE(random_state=1)],
# clf_xgb = XGBClassifier(objective='binary:logistic',
#                         missing=None,
#                         seed=1)
# clf_xgb.fit(X_train, y_train,
#             verbose=True,
#             early_stopping_rounds=10,
#             eval_metric='aucpr',
#             eval_set=[(X_test, y_test)])

# param_grid_xgb = {'max_depth': [3, 4, 5],
#                   'learning_rate': [0.1, 0.2],
#                   'gamma': [0, 0.1, 0.15],
#                   'reg_lambda': [0, 0.05, 0.1],
#                   'scale_pos_weight': [1, 2, 2.5]}

param_grid_xgb = {'classifier__max_depth': [3, 4, 5, 6],
                  'classifier__learning_rate': [0.1, 0.2, 0.3],
                  'classifier__gamma': [0, 0.1, 0.15],
                  'classifier__reg_lambda': [0.1, 0.2，0.3, 0.4]}

grid_xgb = GridSearchCV(estimator=pipeline_xgb,
                        param_grid=param_grid_xgb,
                        scoring=scoring,
                        n_jobs=-1,
                        cv=stratified_kfold,
                        verbose=1)

grid_xgb.fit(X_train, y_train.values.ravel())

print(grid_xgb.best_params_)
cv_score_xgb = grid_xgb.best_score_
test_score_xgb = grid_xgb.score(X_test, y_test)
xgbpred = grid_xgb.predict(X_test)

print(f'xgb Best Estimator: {grid_xgb.best_estimator_}\n'
      f'xgb Cross-validation score: {cv_score_xgb}\n'
      f'xgb Test score: {test_score_xgb}\n'
      )

print(
    f'xgbAccuracy = {accuracy_score(y_test, xgbpred): .2f}\n'
    f'xgb Recall = {recall_score(y_test, xgbpred): .2f}\n'
    f'xgb Precision = {precision_score(y_test, xgbpred): .2f}\n'
    f'xgb F1 = {f1_score(y_test, xgbpred): .2f}\n'
)

# --------------------------------- 2. Random Forest --------------------------------------
