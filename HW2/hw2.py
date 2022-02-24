import pandas as pd
#import arff
from scipy.io import arff as scipy_arff
import arff

from sklearn.linear_model import LinearRegression
    
def write_arff(filename, df, relation="pima_diabetes"):
    """ Writes data from a df into an arff file. """
    lines = []
    # Add relation
    lines.append("@relation " + relation)

    # Classify attributes by python type, add to lines
    attributes = [(j, 'real') if df[j].dtypes in ['int64', 'float64', 'float', 'int'] \
                   else (j, 'integer') if df[j].dtypes in ['int'] \
                   else (j, 'string') if df[j].dtypes in ['str'] \
                   else (j, '{True, False}') if df[j].dtypes in ['bool'] \
                  else (j, "{" + ", ".join(df[j].unique().astype(str).tolist()) + "}") for j in df]
    for a in attributes:
        lines.append("@attribute " + a[0] + " " + a[1])

    # Add the data points
    lines.append("@data")
    for row in df.values:
        lines.append(",".join([i.decode() if type(i) == bytes else str(i) for i in row]))

    open(filename, "w").writelines([l + "\n" for l in lines])


def test_regression(df, target, feature):
    """ Returns the result of linear regression using feature to predict target, removes 0's"""
    regressor = LinearRegression()
    test_df = df[[target, feature]]
    test_df = test_df[(test_df[target] > 0) & (test_df[feature] > 0)]
    X, Y = test_df[[feature]], test_df[[target]]
    regressor.fit(X, Y)
    #y_pred = regressor.predict(X)
    coef, intercept = regressor.coef_, regressor.intercept_
    score = regressor.score(X, Y)
    return coef, intercept, score

def test_multiregression(df, target, features):
    """ Returns the result of linear regression using features to predict target. """
    training_df = df[df[target] > 0][features + [target]]
    target_df = training_df[[target]]
    training_df.drop(target, axis=1, inplace=True)

    lm = LinearRegression()
    model = lm.fit(training_df, target_df)
    coef, intercept, score = model.coef_, model.intercept_, model.score(training_df, target_df)
    return lm, coef, intercept, score


def convert_features_to_bool(df, features):
    """ Given a df, return a copy where a column has been turned into bools, df[feature] > 0 => 1 """
    df = df.copy()
    df[features] = df[features].astype(bool)
    return df


def fill_with_mean(df):
    """ Given a df, return a copy where blanks have been filled with the mean for their column. """
    return df.replace(0, df.mean(axis=0))

def fill_with_median(df):
    """ Given a df, return a copy where blanks have been filled with the median for their column. """
    return df.replace(0, df.median(axis=0))

def fill_column_with_regression(df, target, features):
    """ Given a df, return a copy with 0's replaced by linear regression using features  """
    print("\nReplacing null-values in dataframe with linear regression. Target: ", target, "Features: ", features)
    df = df.copy()
    model, coef, intercept, score = test_multiregression(df, target, features)
    print("Coef: ", coef, "\nIntercept: ", intercept, "\nScore: ", score)
    df[target].mask(df[target] <= 0, model.predict(df[features]), inplace=True)
    return df


data = scipy_arff.loadarff('diabetes.arff')

df = pd.DataFrame(data[0])

training_df = df[(df['mass'] > 0) & (df['pres'] > 0) & (df['plas'] > 0) & (df['skin'] > 0) & (df['insu'] > 0)]
#training_df.drop('class', axis=1, inplace=True)

testing_df = df[(df['mass'] <= 0) | (df['pres'] <= 0) | (df['plas'] <= 0) | (df['skin'] <= 0) | (df['insu'] <= 0)]

write_arff('diabetes_training.arff', training_df)
write_arff('diabetes_testing.arff', testing_df)

print("\nCorrelations between presence of data points")
booled_df = convert_features_to_bool(df, [i for i in df.columns.values.tolist() if i != 'class'])
print(booled_df.corr())
print("\Correlation/nRandomness of skin and insulin")
print(booled_df.corr()[['skin', 'insu']])
print("Therefore, as we see, the presence or lack thereof of skin and insulin data points are correlated at 66%")

print("\n Correlations between data points")
print(df.corr())
print("\n Correlations between data points (all 0's removed)")
print(training_df.corr())

print("\n\n Creating df with data in skin/insu binarized by presence")
write_arff('diabetes_skin_insu_binarized.arff', convert_features_to_bool(df, ['skin', 'insu']))

print("\n\n Creating df with 0's replaced by mean column values ")
write_arff('diabetes_mean.arff', fill_with_mean(df))

print("\n\n Creating df with 0's replaced by median column values ")
write_arff('diabetes_median.arff', fill_with_median(df))

print("\n\n Calculating regression factors using other data points")
for t in ['skin', 'insu', 'mass', 'plas', 'pres']:
    print("\n\n TESTING ", t, [i for i in df.columns.values.tolist() if i not in ['class', t]])
    model, coef, intercept, score = test_multiregression(df, t, [i for i in df.columns.values.tolist() if i not in ['class', t]])
    print("Coef: ", coef, "\nIntercept: ", intercept, "\nScore: ", score)


print ("\n\n Replace the 0's with values based on linear regression")
regress_df = df.copy()
df2 = fill_column_with_regression(df, 'skin', ['mass', 'pedi'])
regress_df['skin'] = df2['skin']
df3 = fill_column_with_regression(df, 'insu', ['preg', 'plas', 'pres', 'mass', 'pedi', 'age'])
regress_df['insu'] = df2['insu']
df4 = fill_column_with_regression(df, 'mass', ['skin', 'pedi'])
regress_df['mass'] = df2['mass']
df5 = fill_column_with_regression(df, 'plas', ['skin', 'mass', 'pedi', 'age'])
regress_df['plas'] = df2['plas']
df5 = fill_column_with_regression(df, 'pres', ['mass', 'pedi', 'age'])
regress_df['pres'] = df2['pres']
                                  
write_arff('diabetes_regression.arff', regress_df)
