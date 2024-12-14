import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MultiLabelBinarizer
import ast
def main():
    df = pd.read_csv('recipes.csv')

    df['ingredient_ids'] = df['ingredient_ids'].apply(ast.literal_eval)

    X = df.drop(columns=['id','sodium'])
    y = df['sodium']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    print('step2')
    model = LinearRegression(n_jobs=-1)
    mlb = MultiLabelBinarizer()
    mlb.fit(X_train['ingredient_ids'])

  

    preprocessor = ColumnTransformer(
        transformers=[
            ('binarize_ingredient_ids', FunctionTransformer(mlb.transform, validate=False), 'ingredient_ids'),
            ('scale_numeric', StandardScaler(), ['fat', 'protein', 'calories', 'saturated_fat', 'sugar'])
        ]
    )

    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ]
    )
    print('step1')


    print('more step')
    pipeline.fit(X_train, y_train)
    print('step3')

    with open('linear_regression.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(opt_mse)
    param_dist = {'fit_intercept': [True, False], 'normalize': [True, False]}
    opt = RandomizedSearchCV(
        pipeline, 
        param_dist,   
        n_iter=10,
        cv=5,  
        n_jobs=1,
        verbose=3
    )
    opt.fit(X_train, y_train)
    print(opt.score(X_test, y_test))
    y_pred = opt.predict(X_test)
    opt_mse = mean_squared_error(y_test, y_pred)
    print(opt_mse)
    with open('linear_regression_optimized.pkl', 'wb') as f:
        pickle.dump(opt.best_estimator_, f)
    return mse, opt_mse
print(main())
