# -----------------------------------------------------------
# Exampaple of deeplearning regression problem. Starting from raw data
#
#
# The model is created usgin tensorflow + + pandas  keras + numpy
#
# dmike16, Rome, Italy
# Released under MIT license
# email cipmiky@gmail.com
# -----------------------------------------------------------

import seaborn as sns

import bag.auto_mpg as ampg

sns.set_theme()
# download the data set
autompg = ampg.AutoMPG()
# inspect the dataset with pandas
print(">>>>>>>>>>>>>>>")
autompg.tail()
print(">>>>>>>>>>>>>>>")
autompg.checl_for_unknow_values()
# cleanup and format the data
autompg.cleanup_data()
print(">>>>>>>>>>>>>>>")
autompg.tail()
# split the date
train, test = autompg.split(0.8)
train_ds, test_ds = ampg.AutoMPG.to_dataset(train, test)
# plot the data
# sns.pairplot(train_data[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
# plt.show()
ampg.AutoMPG.statistics(train)
