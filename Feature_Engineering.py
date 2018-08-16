def model(all_data):
    import pandas as pd
    from scipy.stats import norm, skew
    
    #Transforming some numerical variables that are really categorical
    # MSSubClass=The building class,OverallCond
    # Year and month sold are transformed into categorical features.
    for col in ('mssubclass','overallcond','yrsold','mosold'):
         all_data[col] = all_data[col].apply(str)

    # Label Encoding some categorical variables that may contain information in their ordering set
    cols = ('fireplacequ', 'bsmtqual', 'bsmtcond', 'garagequal', 'garagecond', 
            'exterqual', 'extercond','heatingqc', 'poolqc', 'kitchenqual', 'bsmtfintype1', 
            'bsmtfintype2', 'functional', 'fence', 'bsmtexposure', 'garagefinish', 'landslope',
            'lotshape', 'paveddrive', 'street', 'alley', 'centralair', 'mssubclass', 'overallcond', 
            'yrsold', 'mosold')
    # process columns, apply LabelEncoder to categorical features
    for i in cols:
        all_data[i] = all_data[i].astype('category').cat.codes

    # Adding one more important feature
    # Since area related features are very important to determine house prices, we add one more feature which is the total area
    # of basement, first and second floor areas of each house
    # Adding total sqfootage feature 
    all_data['totalsf'] = all_data['totalbsmtsf'] + all_data['1stflrsf'] + all_data['2ndflrsf']
    
    # Skewed features
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    
    # Applying Box Cox Transformation of (highly) skewed features
    skewness = skewness[abs(skewness) > 0.75]
    from scipy.special import boxcox1p
    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        all_data[feat] = boxcox1p(all_data[feat], lam)
    
    #For the remaining categorical variables get dummy categorical features
    all_data = pd.get_dummies(all_data)
    
    return all_data
