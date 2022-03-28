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

##### 1. Check for missing data
```
pct_missing = df.isnull().mean().round(4).mul(100).sort_values(ascending=False)  # no missing
print(pct_missing)
```
There is no missing data points in this dataset

##### 2. Check for imbalanced data
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
- This dataset is imbalanced
<img src="https://user-images.githubusercontent.com/86807275/160313634-759aa796-7f0f-40ca-b470-06264a4620b9.png" width="600" height="500">

##### 3. Format modification
- Change payment indicators, sex, education, marriage status to categorical variable
- Need to drop customer ID as it does not contain any useful information
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
##### 4. Preliminary feature engineering
- compute utilization rate, which is a good indicator of the borrower's account status
- compute utilization rate increase indicator - continuously increasing utilization rate signals financial stress
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
##### 5. Cap and floor certain variables
- Floor negative bill balances (bill credits) at 0
- Floor negative utilization rates at 0
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
  - Univarant analysis graphs - (Numerical variables -> Box plots, Distribution plots | Categorical variables -> Bar plots 
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

<details><summary>Click to view univariant analysis graphs</summary>
<p>

#### Below is resulting univarant analysis graphs - (Numerical variables -> Box plots, Distribution plots | Categorical variables -> Bar plots  


![AGEbox_plot](https://user-images.githubusercontent.com/86807275/160311805-83412b37-0d54-4a41-823b-c0bcf7a42ab2.png)
![AGEdist_plot](https://user-images.githubusercontent.com/86807275/160311824-23df68a0-3e40-480f-b81b-f9b836f5129f.png)
![BILL_AMT1box_plot](https://user-images.githubusercontent.com/86807275/160311933-a06d025b-681e-48d4-8e19-f2ce33238551.png)
![BILL_AMT1dist_plot](https://user-images.githubusercontent.com/86807275/160311940-3e4c51d7-450f-4047-8f36-8fed03aa10e4.png)
![BILL_AMT2box_plot](https://user-images.githubusercontent.com/86807275/160311944-20b33e87-0b42-44de-b50a-8f15276b84e4.png)
![BILL_AMT2dist_plot](https://user-images.githubusercontent.com/86807275/160311960-3da4fc19-a788-4f67-849d-cd0f03c14396.png)
![BILL_AMT4box_plot](https://user-images.githubusercontent.com/86807275/160312144-1a541266-d505-42c6-b4bf-d0714fab463c.png)
![BILL_AMT4dist_plot](https://user-images.githubusercontent.com/86807275/160312147-2f2c11e1-d7a3-45a2-a067-6ec4f0861003.png)
![BILL_AMT5box_plot](https://user-images.githubusercontent.com/86807275/160312149-199eec43-fc2d-4890-a104-9c5bbcab7cb1.png)
![BILL_AMT5dist_plot](https://user-images.githubusercontent.com/86807275/160312150-90de3b13-10c8-4d53-9641-5ba6db7b2d82.png)
![BILL_AMT6box_plot](https://user-images.githubusercontent.com/86807275/160312151-286067da-c1f2-4b1c-8deb-7ac73cdce487.png)
![BILL_AMT6dist_plot](https://user-images.githubusercontent.com/86807275/160312152-dbfecfff-267c-4059-a746-ad51f1e0aead.png)
![cons_util_incbox_plot](https://user-images.githubusercontent.com/86807275/160312153-d3ee4acb-1652-43ff-b483-fc16c5468484.png)
![cons_util_incdist_plot](https://user-images.githubusercontent.com/86807275/160312154-021f5cf9-b5c0-4e87-8e1d-9c40522737a3.png)
![EDUCATION_pct_plot](https://user-images.githubusercontent.com/86807275/160312155-d14e486c-bc6a-4c9f-a796-9308f480d137.png)
![LIMIT_BALbox_plot](https://user-images.githubusercontent.com/86807275/160312156-82edf534-6fd5-4cde-b5a5-b872ca6262c1.png)
![LIMIT_BALdist_plot](https://user-images.githubusercontent.com/86807275/160312157-3779e832-0bf9-4927-82f8-80fbbd6010d0.png)
![MARRIAGE_pct_plot](https://user-images.githubusercontent.com/86807275/160312158-335b075e-e720-4c1f-a957-17b7caaca65d.png)
![PAY_0box_plot](https://user-images.githubusercontent.com/86807275/160312159-2f7b14e1-4efc-496f-b704-2c597065b33e.png)
![PAY_0dist_plot](https://user-images.githubusercontent.com/86807275/160312160-65ec94e7-fe9f-4c0f-b884-443c763c5b90.png)
![PAY_2box_plot](https://user-images.githubusercontent.com/86807275/160312161-cfdf3231-408b-4e77-b3d8-6d9878f81884.png)
![PAY_2dist_plot](https://user-images.githubusercontent.com/86807275/160312162-3ec1241f-9603-4ea8-bc0f-4b749db59057.png)
![PAY_3box_plot](https://user-images.githubusercontent.com/86807275/160312163-c0afcdd5-21bf-4158-baf0-bb0083f5728f.png)
![PAY_3dist_plot](https://user-images.githubusercontent.com/86807275/160312164-32d313de-1649-4bf1-97f6-c8c3beadf690.png)
![PAY_4box_plot](https://user-images.githubusercontent.com/86807275/160312165-95154076-6aa5-48cd-af8d-fd3a20fb2acd.png)
![PAY_4dist_plot](https://user-images.githubusercontent.com/86807275/160312168-0c9e1494-dcce-4a9d-8816-77501ee75cde.png)
![PAY_5box_plot](https://user-images.githubusercontent.com/86807275/160312169-1bc8aba4-3a57-48ce-b679-05a3cc35fb4b.png)
![PAY_5dist_plot](https://user-images.githubusercontent.com/86807275/160312171-bc25e422-b3c2-46c6-bfe7-d441299a2717.png)
![PAY_6box_plot](https://user-images.githubusercontent.com/86807275/160312172-6a51e7a0-b6d8-40c8-a028-dcec78eb2d24.png)
![PAY_6dist_plot](https://user-images.githubusercontent.com/86807275/160312175-e297e84a-b3e0-4e8d-8e7d-b1f1cbcf704d.png)
![PAY_AMT1box_plot](https://user-images.githubusercontent.com/86807275/160312176-05fa6006-ee9d-43ec-af36-89a16ee76871.png)
![PAY_AMT1dist_plot](https://user-images.githubusercontent.com/86807275/160312178-69cadb5f-041b-4cd9-a250-cfc5840e8e20.png)
![PAY_AMT2box_plot](https://user-images.githubusercontent.com/86807275/160312179-dbe9fc86-d8de-4d31-9857-52f53bb7f3fe.png)
![PAY_AMT2dist_plot](https://user-images.githubusercontent.com/86807275/160312180-a1302b02-4ee7-4582-bbd0-7255ed8abf0e.png)
![PAY_AMT3box_plot](https://user-images.githubusercontent.com/86807275/160312181-7c099dd2-8431-4d4d-a455-b03b1c937476.png)
![PAY_AMT3dist_plot](https://user-images.githubusercontent.com/86807275/160312182-92bbe5f8-0aae-4c89-aa92-1fccd2ea08c6.png)
![PAY_AMT4box_plot](https://user-images.githubusercontent.com/86807275/160312184-42260718-0099-4b2c-8681-086c7c486eed.png)
![PAY_AMT4dist_plot](https://user-images.githubusercontent.com/86807275/160312185-9e8b0144-f77c-49ca-b69f-2ab4b5fff2d7.png)
![PAY_AMT5box_plot](https://user-images.githubusercontent.com/86807275/160312186-6eb210fe-d1d7-4e3b-860b-fca6004b1175.png)
![PAY_AMT5dist_plot](https://user-images.githubusercontent.com/86807275/160312189-5f9ce32b-14f8-4315-af89-407551d55c03.png)
![PAY_AMT6box_plot](https://user-images.githubusercontent.com/86807275/160312190-22981ee7-a813-426c-859c-c914e117a0ee.png)
![PAY_AMT6dist_plot](https://user-images.githubusercontent.com/86807275/160312191-90a2fd1e-bfd9-4779-9ed1-f5432ca12fd1.png)
![SEX_pct_plot](https://user-images.githubusercontent.com/86807275/160312193-ddbf564b-c4a6-4ea8-b57a-dcef17a82b9b.png)
![UTIL1box_plot](https://user-images.githubusercontent.com/86807275/160312194-64f0021f-9e5b-4415-bcc8-12bcb85dc9e0.png)
![UTIL1dist_plot](https://user-images.githubusercontent.com/86807275/160312195-9499b51d-eaaa-4fe0-831a-14f632e25065.png)
![UTIL2box_plot](https://user-images.githubusercontent.com/86807275/160312197-c7d7f769-6f0d-434b-84ac-d638337eb7b9.png)
![UTIL2dist_plot](https://user-images.githubusercontent.com/86807275/160312198-15422372-e950-42da-9413-372a05373722.png)
![UTIL3box_plot](https://user-images.githubusercontent.com/86807275/160312200-bffd3c14-2ce3-4b27-8c4a-04c5fcbff0f7.png)
![UTIL3dist_plot](https://user-images.githubusercontent.com/86807275/160312201-b00841a9-3ad7-422c-ad58-91bc3ec60a24.png)
![UTIL4box_plot](https://user-images.githubusercontent.com/86807275/160312202-84adba56-1f4f-4734-9cc0-560ec3e8b5dd.png)
![UTIL4dist_plot](https://user-images.githubusercontent.com/86807275/160312204-97953216-f047-4320-90e1-847e37b073ed.png)
![UTIL5box_plot](https://user-images.githubusercontent.com/86807275/160312205-e9fb7e99-bce7-45c5-b1d8-21064d7587fb.png)
![UTIL5dist_plot](https://user-images.githubusercontent.com/86807275/160312206-a2ee78fd-3c94-4b81-861a-d378d21b119e.png)
![UTIL6box_plot](https://user-images.githubusercontent.com/86807275/160312207-1cc16db8-13a7-4d2b-9b45-c46886ca1ebf.png)
![UTIL6dist_plot](https://user-images.githubusercontent.com/86807275/160312208-18089d82-8aea-47d6-93ae-ea32f6912312.png)

    ```

</p>
</details>

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

