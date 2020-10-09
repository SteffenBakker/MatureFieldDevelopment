import numpy, scipy.io
import os
import pandas as pd
import datetime


#os.chdir('C:\\Users\\name\\folders') #get the correct working directory
remove_data_up_to_year = 2006
num_years = 10
num_contracts = 20

#READ DATA
data = pd.read_excel('S&S.xlsx', sheet_name='Futures2')
data = data.fillna('-') # with 0s rather than NaNs
data['Year'] =  data['Month'].map(lambda x: x.year)

prices = pd.read_excel('S&S.xlsx', sheet_name='Spot2')
prices['Year'] =  prices['Start'].map(lambda x: x.year)

#SUBSETTING
data = data[data['Year'] >= remove_data_up_to_year]
prices = prices[prices['Year'] >= remove_data_up_to_year]

remove_years = numpy.arange(2000,remove_data_up_to_year)
remove_months = numpy.arange(1,12+1)
for year in remove_years:
    for month in remove_months:
        if datetime.datetime(year, month, 1, 0, 0) in list(data.columns):
            del data[datetime.datetime(year, month, 1, 0, 0)]

#FORMATTING
prices['YearMonth'] = [d.strftime("%Y/%m") for d in prices['Start']]
prices = prices[['YearMonth','SpotPrice']]
prices.reset_index()
prices = prices.groupby(['YearMonth'],as_index=False).mean()
prices = prices.dropna() #remove the last rows.
prices.rename(columns={'SpotPrice':0}, inplace=True)

data['YearMonth'] = [d.strftime("%Y/%m") for d in data['Month']]   # data['Month'].format
columns = [c for c in list(data.columns) if c not in ['Year', 'From', 'To',] ]
data = data[columns]
data = data.reset_index()

def diff_month(d1, d2):
    return (d2.year - d1.year) * 12 + d2.month - d1.month

data_ = pd.DataFrame(index=list(data.index), columns=numpy.arange(1,12*num_years))
columns = [c for c in list(data.columns) if c not in ['YearMonth','Month','index']]
for row in list(data.index):
    for column in columns:
        if data[column][row] != '-':   # !np.isnan(x)
            month = diff_month(data['Month'][row],column)
            if month > 1:
                data_[month][row] = data[column][row]

for col in data_:
    data_[col] = pd.to_numeric(data_[col], errors='coerce')

#INTERPOLATION
data_ = data_.interpolate(method='linear',axis=1)

#getting data in right format
data_['YearMonth'] = data['YearMonth']
del data_[1]
data_ = data_.dropna() #remove the last rows. CHECK!
data_ = data_.groupby(['YearMonth'],as_index=False).mean()  #reducing the size of the dataset. Monthly observations instead of weekly.
data_ = pd.merge(data_, prices, on='YearMonth')
pick_contracts = [0] + list(numpy.round(numpy.linspace(2,12*num_years-1,num=num_contracts)).astype(int))
data_ = data_[pick_contracts]
data_ = data_.apply(lambda x: numpy.log(x))
oil_contracts = data_.rename_axis('ID').values #finally to array

#maturity input for matlab:
maturities = list(numpy.round(numpy.linspace(2,12*num_years-1,num=num_contracts)).astype(int))
string = ''
for x in [str(m)+'/12,' for m in maturities]:
    print(x,end=" ")

[str(m)+'/12' for m in maturities]

#SAVE_DATA
scipy.io.savemat('oil_future_contracts.mat', mdict={'oil_contracts': oil_contracts})