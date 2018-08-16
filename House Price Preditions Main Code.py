import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from scipy import stats


### Read Data ##
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
train.columns = [c.lower() for c in train.columns]
test.columns = [c.lower() for c in test.columns]
sub_index = test['id']
train.drop(columns=['id'],inplace=True)
test.drop(columns=['id'],inplace=True)

# Deleting outliers as identified by EDA
train.drop(train[(train['grlivarea']>4000) & (train['saleprice']<300000)].index,inplace=True)

train_objs_num = len(train)
y = train['saleprice']
dataset = pd.concat(objs=[train.drop(columns=['saleprice']), test], axis=0, ignore_index=True)
all_data = dataset.copy()
all_data.shape

import Clean,Feature_Engineering,Simple_Stacking
all_data = Clean.model(all_data,train_objs_num)
all_data = Feature_Engineering.model(all_data)

## Apply Log tranformation to target variable as it is right skewed
y = np.log1p(y)
f_train = all_data[:train_objs_num]
f_test  = all_data[train_objs_num:]

predictions = Simple_Stacking.model(f_train,y,f_test)

final_predictions = predictions

submission = pd.DataFrame({ 'Id': sub_index,
                            'SalePrice': final_predictions.astype(float)})
submission.to_csv("submission.csv", index=False)

final_predictions.astype(float)
