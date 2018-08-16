def model(X,y,z):
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn import linear_model
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    import xgboost as xgb
    from sklearn.model_selection import RandomizedSearchCV
    from xgboost import XGBRegressor
    from sklearn.metrics.scorer import make_scorer
    
    def rmse_eval(y, y0):
        error = np.sqrt(np.mean(np.power(y-y0, 2)))
        return error
    my_scorer = make_scorer(rmse_eval, greater_is_better=True)
    par_rf = {'n_estimators': [100, 150, 200,300],
              'max_depth' : [3,6,9,12]}
    par_dt = {'max_features': ['auto', 'sqrt', 'log2'],
              'max_depth' : [3,6,9,12]}
    par_xg = {'n_estimators': [100, 150, 200,300],
              'max_depth' : [3,6,9,12]}
    
    model1 = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=42), param_distributions=par_rf, cv= 10, n_iter=1,scoring=my_scorer)
    model2 = RandomizedSearchCV(estimator=DecisionTreeRegressor(random_state=42), param_distributions=par_dt, cv= 10, n_iter=1,scoring=my_scorer)
    model3 = RandomizedSearchCV(estimator=xgb.XGBRegressor(random_state=42), param_distributions=par_xg, cv= 5, n_iter=10,scoring=my_scorer)

    # Fit it to the data
    model1.fit(X, y)
    model2.fit(X, y)
    model3.fit(X, y)

    #store preds on test and train data
    preds1 = model1.predict(X)
    preds2 = model2.predict(X)
    preds3 = model3.predict(X)
    test_preds1 = model1.predict(z)
    test_preds2 = model2.predict(z)
    test_preds3 = model3.predict(z)
    print(X.values)
    print(preds1)
    
    #store predictions
    stacked_predictions = np.column_stack((preds1,preds2,preds3))
    stacked_test_predictions = np.column_stack((test_preds1,test_preds2,test_preds3))
    print(stacked_predictions)
    #Fit & predict with the meta model
    meta_model = linear_model.LinearRegression()
    meta_model.fit(stacked_predictions,y)
    final_predictions = np.expm1(meta_model.predict(stacked_test_predictions))
    df2 = pd.DataFrame(data=[])
    df2['true'] = np.expm1(meta_model.predict(stacked_predictions))
    df2['pred_rf'] = np.expm1(y)
    df2[['true','pred_rf']].plot()
    print('Train score of model1:',rmse_eval(y, preds1))
    print('Train score of model2:',rmse_eval(y, preds2))
    print('Train score of model3:',rmse_eval(y, preds3))
    print('Train score of stacked model:',rmse_eval(y, meta_model.predict(stacked_predictions)))
    return final_predictions