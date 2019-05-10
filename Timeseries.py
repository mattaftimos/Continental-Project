import datetime
import pandas as pd
import numpy as np
import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy
import numpy as np
#group for cost centers
#for each row in that group pull out the startdate and end date, make it into absent date for each day
#then group by cost centers and absent date then count rows then create new file with cost center absent date and count of absentees

absences_df = pd.read_pickle('cleaneddata/absences.dat')
absences_df['date'] = np.NAN
newdata = []
absences_One = absences_df.groupby('Cost Center')

date_format = "%Y/%m/%d"

for index, row in absences_df.iterrows():
    a = pd.Timestamp.date(row['Start Date'])
    b = pd.Timestamp.date(row['End Date'])
    delta = b-a
    date = a
    for i in range(delta.days):
        row['date'] = date + datetime.timedelta(days=i)
        test = row.values.copy()
        newdata.append(test)


m_absences_df = pd.DataFrame.from_records(newdata,columns=list(absences_df.columns.values))
m_absences_df.to_pickle('cleaneddata/mabsences.dat')
m_absences_df.to_csv('cleaneddata/mabsences.csv')
print(m_absences_df.head(5))



#ask whether 0 days is either half way or no day off etc

g1 = pd.DataFrame({'absentees': m_absences_df.groupby(["Cost Center", "date"]).size()}).reset_index()
g1["Cost Center"] = g1["Cost Center"].str.replace('/', '-')
g2 = g1.groupby(["Cost Center"])
for name in g2["Cost Center"].unique():
    df = pd.DataFrame(g2.get_group(name[0]))
    year1 = df.iloc[0]['date'].year
    year2 = df.iloc[df.shape[0] - 1]['date'].year
    if not (datetime.datetime(year1, 1, 1) in df['date']):
        df = pd.concat(
            [df, pd.DataFrame([[datetime.datetime(year1, 1, 1), 0]], columns=['date', 'absentees'])])

    if not (datetime.datetime(year2, 12, 31) in df['date']):
        df = pd.concat(
            [df, pd.DataFrame([[datetime.datetime(year2, 12, 31), 0]], columns=['date', 'absentees'])])
    # df = pd.concat(
    #     [df, pd.DataFrame([[datetime.datetime(year2+1, 12, 31), 0]], columns=['date', 'absentees'])])
    df.index = pd.to_datetime(df["date"])
    df = df.resample('D').replace(np.nan, 0)
    df = df.drop(df[(df.index.day == 29) & (df.index.month == 2)].index)
    df = df.drop(df[(df.index.year < 2016)].index)

    df.to_pickle('cleaneddata/Timeseries/'+name[0]+'.dat')
    df.to_csv('cleaneddata/Timeseries/'+name[0]+'.csv')