def model(all_data,train_objs_num):
    import pandas as pd
    # PoolQC : data description says NA means "No Pool". 
    # MiscFeature : data description says NA means "no misc feature"
    # Alley : data description says NA means "no alley access"
    # Fence : data description says NA means "no fence"
    # FireplaceQu : data description says NA means "no fireplace"
    # GarageType, GarageFinish, GarageQual and GarageCond : Replacing missing data with None
    # BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2 : For all these categorical
    # basement-related features,NaN means that there is no basement.
    # MSSubClass : Na most likely means No building class. We can replace missing values with None
    # MasVnrArea and MasVnrType : NA most likely means no masonry veneer for these houses. We can fill
    # 0 for the area and None for the type.
    for col in ('mssubclass','masvnrtype','bsmtqual', 'bsmtcond', 'bsmtexposure', 'bsmtfintype1', 'bsmtfintype2','fireplacequ','poolqc','miscfeature','alley','fence','garagetype', 'garagefinish', 'garagequal', 'garagecond'):
        all_data[col] = all_data[col].fillna('None')
    
    # LotFrontage : Since the area of each street connected to the house property most likely have a similar
    # area to other houses in its neighborhood , we can fill in missing values of LotFrontage based on the 
    # median of LotArea and Neighborhood. Since LotArea is a continuous feature, We use qcut to divide it 
    # into 10 parts.
    all_data["lotfrontage"] = all_data.groupby("neighborhood")["lotfrontage"].transform(lambda x: x.fillna(x.median()))

    # GarageYrBlt, GarageArea and GarageCars : Replacing missing data with 0 (Since No garage = no cars in such garage.)
    # BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath : missing values are likely zero for having
    # no basement
    for col in ('masvnrarea','bsmtfinsf1', 'bsmtfinsf2', 'bsmtunfsf', 'totalbsmtsf', 'bsmtfullbath', 'bsmthalfbath','garageyrblt', 'garagearea', 'garagecars'):
        all_data[col] = all_data[col].fillna(0)

    # MSZoning (The general zoning classification) : So we can fill in missing values with the most common value
    # Electrical : It has one NA value. So we can fill in missing values with the most common value
    # KitchenQual: Only one NA value, and same as Electrical, we fill in missing values with the most common value
    # Exterior1st and Exterior2nd : Again Both Exterior 1 & 2 have only one missing value. we fill in  with the most common value
    # SaleType : Fill in again with most frequent value
    for col in ('mszoning','electrical','kitchenqual','exterior1st','exterior2nd','saletype'):
        all_data[col] = all_data[col].fillna(all_data[:train_objs_num][col].mode()[0])

    # Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with
    # 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it.
    all_data = all_data.drop(['utilities'], axis=1)

    # Functional : data description says NA means typical
    all_data["functional"] = all_data["functional"].fillna("Typ")
    
    return all_data