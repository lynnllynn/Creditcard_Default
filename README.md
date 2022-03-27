## Introduction
Purpose of this notebook is to construct a default prediction model on credit card clients

## Dataset Description
This dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.

There are 25 variables:

* ID: ID of each client
* LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
* SEX: Gender (1=male, 2=female)
* EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
* MARRIAGE: Marital status (1=married, 2=single, 3=others)
* AGE: Age in years
* PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, … 8=payment delay for eight months, 9=payment delay for nine months and above)
* PAY_2: Repayment status in August, 2005 (scale same as above)
* PAY_3: Repayment status in July, 2005 (scale same as above)
* PAY_4: Repayment status in June, 2005 (scale same as above)
* PAY_5: Repayment status in May, 2005 (scale same as above)
* PAY_6: Repayment status in April, 2005 (scale same as above)
* BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
* BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
* BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
* BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
* BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
* BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
* PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
* PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
* PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
* PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
* PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
* PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
* default.payment.next.month: Default payment (1=yes, 0=no)

## Pre-model processes
### Load packages 
```
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
```
### Data Cleaning and pre-processing

1. Check for missing data
```
pct_missing = df.isnull().mean().round(4).mul(100).sort_values(ascending=False)  # no missing
print(pct_missing)
```
2. Check for imbalanced data
```
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
```
3. Format modification
```
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

```
4. Preliminary feature engineering
```
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
```
5. Cap and floor certain variables
```
# Floor negative bill balances (bill credits) at 0
# Floor negative utilization rates at 0
for i in range(1, 7):
    bill = 'BILL_AMT' + str(i)
    util = 'UTIL' + str(i)
    df[util] = df[util].apply(lambda x: 0 if (x < 0) else x)
    df[bill] = df[bill].apply(lambda x: 0 if (x < 0) else x)

print(df.describe())
```

### Data Exploration
1. Preliminary Screening - Uni-variate and Bi-variate Analysis
  - Compute Point Bi-serial Correlation for Categorical Y vs Continuous X
  - Compute Chi-squared statistics for Categorical Y vs Categorical
```
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
```
2. Filter Multicollinearity
```
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
```
3. Outlier filter
```
# Detect and drop outliers using 3 std threshold
print(' Before removing outliers', df.shape)
df = df[(np.abs(stats.zscore(df.select_dtypes(include=[np.number])) < 3).all(axis=1))]
print(' after removing outliers', df.shape)
```

### More data pre-processing
Create dummy variables and drop first for categorical data to avoid perfect multicollinearity
```
df_final = pd.get_dummies(df, columns=df_cat.columns, drop_first=True)
```
## Model Building
### Train Test Slpit
```
X = df_final.drop(['def'], axis=1)
y = df_final[['def']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Check percentage of default event
print(y_train['def'].sum() / len(y_train))
print(y_test['def'].sum() / len(y_test))

# print(y[y.isnull().any(axis=1)])
# print(X[X.isnull().any(axis=1)])
```
### Models
Model scoring meterics selection:
scoring = 'roc_auc'

Modeling pipeline:
1. Use SMOTE
  - Imbalance data, 22.7% default rate
3. Standardize numeric attributes
  - Make logistic regression coefficient more interpretable
  - Although logistic does not require homoscedasticity and normality assumptions for residuals, they are still essential for regularization
4. Model hyperparameter tuning
5. Model fitting and selection

#### 1. XGBoost
```
```
#### 2. Random Forest
```
```
#### 3. Logistic Regression
```
```
## Acknowledgements
Any publications based on this dataset should acknowledge the following:

Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

The original dataset can be found here at the UCI Machine Learning Repository.
