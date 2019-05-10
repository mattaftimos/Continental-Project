import datetime
import pandas as pd
import numpy as np


mySuccess_df = pd.read_csv('data/UNCC My Success.csv')
absences_df = pd.read_excel('data/171023 2017 YTD Absences.xlsx', sheet_name='Sheet1')
absences_df2 = pd.read_excel('data/Absences 2016.xlsx', sheet_name='Sheet1')
kanexa_df = pd.read_excel('data/UNCC Kanexa.xlsm.xlsx', sheet_name='R-Closed,Cancelled')
termination_df = pd.read_csv('data/UNCC_Termination 2017.csv')
master_df = pd.read_csv('data/UNCC_HR Master Data clean.csv')

# Cleaning My Success
mySuccess_df[['Have you worked abroad?', 'Have you worked in more than one BU level organization?',
              'Do you have working experience in more than one Functional Area?', 'New to Position',
              'Ready to move (in next 12 months)', 'Goals Applicable', 'Expatriate']] = mySuccess_df[
    ['Have you worked abroad?', 'Have you worked in more than one BU level organization?',
     'Do you have working experience in more than one Functional Area?', 'New to Position',
     'Ready to move (in next 12 months)', 'Goals Applicable', 'Expatriate']].apply(lambda x: x.astype(bool))
mySuccess_df[['Level in Organization', 'Continental Grade', 'Legal Entity (Code)', 'User Sys ID', 'Employee Id']] = \
mySuccess_df[
    ['Level in Organization', 'Continental Grade', 'Legal Entity (Code)', 'User Sys ID', 'Employee Id']].apply(
    lambda x: x.astype('category'))
mySuccess_df[['Date of Birth', 'Hire Date']] = mySuccess_df[['Date of Birth', 'Hire Date']].apply(
    lambda x: pd.to_datetime(x))
mySuccess_df[mySuccess_df.select_dtypes(['object']).columns] = mySuccess_df.select_dtypes(['object']).apply(
    lambda x: x.astype('category'))

# Cleaning Absences
for col in ['Per No', 'Cost Ctr', 'Cost Center', 'PA', 'Personnel Area', 'Employee Group',
            'PSubarea', 'Personnel Subarea', 'A/AType', 'Attendance or Absence Type',
            'Cost Ctr.1']:
    absences_df[col] = absences_df[col].astype('category')

absences_df['Start Date'] = pd.to_datetime(absences_df['Start Date'], infer_datetime_format=True)
absences_df['End Date'] = pd.to_datetime(absences_df['End Date'], infer_datetime_format=True)
absences_df['Hrs'] = pd.to_numeric(absences_df['Hrs'], downcast='float')


absences_df2.rename(columns={'Payroll hrs': 'Hrs', 'Personnel No.': 'Per No'}, inplace=True)

for col in ['Per No', 'Cost Ctr', 'Cost Center', 'PA', 'Personnel Area', 'Employee Group',
            'PSubarea', 'Personnel Subarea', 'Attendance or Absence Type']:
    absences_df2[col] = absences_df2[col].astype('category')

absences_df2['Start Date'] = pd.to_datetime(absences_df2['Start Date'], infer_datetime_format=True)
absences_df2['End Date'] = pd.to_datetime(absences_df2['End Date'], infer_datetime_format=True)
absences_df2['Hrs'] = pd.to_numeric(absences_df2['Hrs'], downcast='float')

# Cleaning Kanexa

for col in ['Auto req ID', 'Country', 'Continental Location', 'Division', 'Cost Center', 'Job Title',
            'Business Unit', 'Internal Position Title', 'Salary Classification', 'Salary Type',
            'Current Req Status', 'Recruiter GID', 'Original', 'Job Type',
            'Salary classification for internal posting']:
    kanexa_df[col] = kanexa_df[col].astype('category')

for col in ['ePR Approved', 'Date Open', 'Written Offer Accepted', 'Date Closed', 'Date Canceled',
            'Started/Hired', 'Date on Hold']:
    kanexa_df[col] = pd.to_datetime(kanexa_df[col], infer_datetime_format=True)

for col in ['ePR Approved', 'Date Open', 'Written Offer Accepted', 'Date Closed', 'Date Canceled',
            'Started/Hired', 'Date on Hold']:
    kanexa_df[col] = pd.to_datetime(kanexa_df[col], infer_datetime_format=True)

# Cleaning Term
for col in ['Personnel No.', 'Action type', 'Personnel Area', 'Personnel Subarea',
            'Reason for action', 'Leaving date', 'Cost Ctr', 'Cost Center', 'Functional Area',
            'Gender Key', 'Birth date', 'Job']:
    termination_df[col] = termination_df[col].astype('category')

termination_df['Entry'] = pd.to_datetime(termination_df['Entry'], infer_datetime_format=True)
termination_df['Chngd on'] = pd.to_datetime(termination_df['Chngd on'], infer_datetime_format=True)
termination_df['Leaving date'] = pd.to_datetime(termination_df['Leaving date'], infer_datetime_format=True)
termination_df['Birth date'] = pd.to_datetime(termination_df['Birth date'], infer_datetime_format=True)

# Cleaning masterdata

for col in ['Pers.No.', 'local ID', 'GID', 'CoCd (Number)', 'Company Code (Text)', 'Personnel Area Code',
            'Personnel Area Name', 'Personnel Subarea Code', 'Personnel Subarea Name', 'Country',
            'Work Contract (Salary Type)', 'Organizational Unit Code', 'Organizational Unit Name', 'Division Name',
            'Full Time Equivalent', 'Monthly Working Hours', 'Weekly Working Hours', 'Capacity Utilization Level',
            'Contract Type', 'Contract End Date', 'Employee Subgroup', 'Gender', 'Nationality', 'Functional Area',
            'Functional Area.1', 'Conti Grade', 'Cost Center', 'Cost Center Name', 'Position', 'Position Name', 'Job',
            'Job Name', 'Function code', 'Function Name', 'Windows User', 'GID of Direct Leader',
            'GID of HR Responsible']:
    master_df[col] = master_df[col].astype('category')

master_df['Service Date'] = pd.to_datetime(master_df['Service Date'])
master_df['Date of Birth'] = pd.to_datetime(master_df['Date of Birth'])
master_df['Nationality'] = master_df["Nationality"].str.strip(to_strip=None)

master_df = master_df.replace('', np.nan, regex=True)
termination_df = termination_df.replace('', np.nan, regex=True)
absences_df = absences_df.replace('', np.nan, regex=True)
mySuccess_df = mySuccess_df.replace('', np.nan, regex=True)
kanexa_df = kanexa_df.replace('', np.nan, regex=True)

master_df = master_df.rename(index=str,
                             columns={"Pers.No.": "PID", "Personnel Area Name": "Personnel Area", "Job": "Job Id",
                                      "Job Name": "Job"})
mySuccess_df = mySuccess_df.rename(index=str, columns={"Employee Id": "PID", "B": "c"})
termination_df = termination_df.rename(index=str, columns={"Gender Key": "Gender", "Personnel No.": "PID",
                                                           "Birth date": "Date of Birth", "Cost Ctr": "Cost Center",
                                                           "Cost Center": "Cost Center Name"})

absences_df = pd.concat([absences_df2,absences_df])

absences_df = absences_df.rename(index=str, columns={"Per No": "PID"})

master_df.dropna(axis=1, how='all')
termination_df.dropna(axis=1, how='all')
absences_df.dropna(axis=1, how='all')
mySuccess_df.dropna(axis=1, how='all')
kanexa_df.dropna(axis=1, how='all')


master_df.to_pickle('cleaneddata/master.dat')
termination_df.to_pickle('cleaneddata/termination.dat')
absences_df.to_pickle('cleaneddata/absences.dat')
mySuccess_df.to_pickle('cleaneddata/mySuccess.dat')
kanexa_df.to_pickle('cleaneddata/kanexa.dat')

master_df = master_df.drop(['Position Name', 'Function code', 'Function Name', 'Windows User'], axis =1)
absences_df = absences_df.drop('Cost Ctr.1', axis = 1)
mySuccess_df = mySuccess_df.drop('Academic Title', axis = 1)

listofdf = [termination_df, mySuccess_df, master_df]
combined_df = pd.concat(listofdf)

combined_df['Gender'] = combined_df['Gender'].str.replace("F","Female")
combined_df['Gender'] = combined_df['Gender'].str.replace("Femaleemale","Female")
combined_df['Gender'] = combined_df['Gender'].str.replace("M","Male")
combined_df['Gender'] = combined_df['Gender'].str.replace("Maleale","Male")


combined_df.to_pickle('cleaneddata/CombinedFile.dat')
combined_df.to_csv('cleaneddata/CombinedFile.csv')

#group for cost centers
#for each row in that group pull out the startdate and end date, make it into absent date for each day
#then group by cost centers and absent date then count rows then create new file with cost center absent date and count of absentees

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



#ask whether 0 days is either half way or no day off etc

g1 = pd.DataFrame({'absentees': m_absences_df.groupby(["Cost Center", "date"]).size()}).reset_index()
g1["Cost Center"] = g1["Cost Center"].str.replace('/', '-')
g2 = g1.groupby(["Cost Center"])
for name in g2["Cost Center"].unique():
    df = pd.DataFrame(g2.get_group(name[0]))
    df.index = pd.to_datetime(df["date"])

    df = df.resample('D').replace(np.nan, 0)
    df.to_pickle('cleaneddata/Timeseries/'+name[0]+'.dat')
