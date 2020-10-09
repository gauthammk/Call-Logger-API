import pandas as pd
# import numpy as np
import datetime
from statistics import stdev
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import confusion_matrix
# from sklearn import metrics
# from sklearn.externals import joblib
import pickle
import json

preMarpreproCallsDf = pd.DataFrame()
postMarpreproCallsDf = pd.DataFrame()
preMarpreproSMSDf = pd.DataFrame()
postMarpreproSMSDf = pd.DataFrame()
relation = {' Friend': 0, ' Work': 1, ' School/College': 2, ' Father': 3, ' Mother': 4, ' Spouse/Partner': 5,
            ' Sister': 6, ' Brother': 7, ' Daughter': 8, ' Son': 9, ' Relative': 10, ' Other': 11, ' Tag Here': 11}
rel = {0: 2, 1: 1, 2: 2, 3: 0, 4: 0, 5: 0,
       6: 0, 7: 0, 8: 0, 9: 0, 10: 3, 11: 4}
mapping = {0: 'Immediate Family', 1: 'Work',
           2: 'Friends', 3: 'Work', 4: 'Other'}
names = dict()


def preproCallData(filename):
    df = pd.read_csv(filename)
    #fn = filename.split('/')
    #username = fn[1].split('_')
    #df['Id'] = df['Id'].apply(lambda x: username[0]+'_'+str(x))
    df[' Date_Time'] = df[' Date_Time'].apply(
        lambda x: datetime.datetime.fromtimestamp(int(x/1000)))
    for index, row in df.iterrows():
        names[row['Id']] = row[' Name']
    df[' Name'] = df[' Name'].apply(lambda x: "null")
    #df[' Relationship'] = df[' Relationship'].apply(lambda x: relationBMS[x])
    Mar = datetime.datetime(2020, 3, 1, 0, 0)
    postMar = df.loc[df[' Date_Time'] >= Mar]
    preMar = df.loc[df[' Date_Time'] < Mar]
    if len(postMar) > 0:
        postMarpreProDf = func2(postMar)
    else:
        postMarpreProDf = None
    if len(preMar) > 0:
        preMarpreProDf = func2(preMar)
    else:
        preMarpreProDf = None
    return postMarpreProDf, preMarpreProDf


def func2(df):
    ids = list(df.Id.unique())
    preproDf = pd.DataFrame(data={'Id': ids})
    TotDur = []
    TotCall = []
    UniqueDayCnt = []
    weekDayCnt = []
    saturDayCnt = []
    sunDayCnt = []
    freq = []
    AvgDuration = []
    SatCallPer = []
    SunCallPer = []
    WeekCallPer = []
    CallsBfrNoon = []
    longTotal = []
    longContact = []
    longWeekDayCallsTotal = []
    longWeekDayCallsContact = []
    maxCountOfWeek = []
    #label = []
    callTypes = {'Incoming': ' 1', 'Outgoing': ' 2'}
    STDTYPE = {'Incoming': [], 'Outgoing': []}
    for i in ids:
        temp = df.loc[df['Id'] == i]
        #rel = list(temp[' Relationship'])
        # label.append(rel[0])
        freq.append(len(temp))
        for Type, j in callTypes.items():
            if(len(temp.loc[df[' CallType'] == j]) >= 2):
                StdCallType = stdev(
                    list(map(int, list(temp.loc[df[' CallType'] == j][' Duration']))))
                STDTYPE[Type].append(StdCallType)
            else:
                STDTYPE[Type].append(0)
        timesOfCalls = list(df.loc[df['Id'] == i][' Date_Time'])
        # .isocalendar().week
        countStor = dict()
        for j in timesOfCalls:
            key = (j.year, j.week)
            if key in countStor:
                countStor[key] = countStor[key]+1
            else:
                countStor[key] = 1
        maxCountOfWeek.append(max(list(countStor.values())))
        TotCall.append(len(temp))
        TotDur.append(sum(temp[' Duration']))
        AvgDuration.append(TotDur[-1]/freq[-1])
        tempdates = list(map(lambda x: x.date(), list(
            df.loc[df['Id'] == i][' Date_Time'])))
        totalNoCalls = len(tempdates)
        weekdays = list(filter(lambda x: x.weekday() !=
                               5 and x.weekday() != 6, tempdates))
        weekDayCnt.append(len(weekdays)*100/totalNoCalls)
        saturdays = list(filter(lambda x: x.weekday() == 5, tempdates))
        saturDayCnt.append(len(saturdays)*100/totalNoCalls)
        sundays = list(filter(lambda x: x.weekday() == 6, tempdates))
        sunDayCnt.append(len(sundays)*100/totalNoCalls)
        uniquedays = len(set(tempdates))
        UniqueDayCnt.append(uniquedays)
        #print(i+": "+"sum: "+str(sum(temp[' Duration'])))
        satDur = 0
        sunDur = 0
        weekDur = 0
        for idx, day in enumerate(tempdates):
            if day in saturdays:
                satDur = satDur+temp.iloc[idx, 4]
            if day in sundays:
                sunDur = sunDur+temp.iloc[idx, 4]
            if day in weekdays:
                weekDur = weekDur+temp.iloc[idx, 4]
        if(TotDur[-1] != 0):
            SatCallPer.append(satDur*100/TotDur[-1])
            SunCallPer.append(sunDur*100/TotDur[-1])
            WeekCallPer.append(weekDur*100/TotDur[-1])
        else:
            SatCallPer.append(0)
            SunCallPer.append(0)
            WeekCallPer.append(0)
        beforenoon = 0
        for time in list(temp[' Date_Time']):
            if time.hour < 12:
                beforenoon = beforenoon+1
        CallsBfrNoon.append(beforenoon*100/freq[-1])
    TotalAvgDuration = sum(TotDur)/len(df)
    preproDf['Frequency'] = freq
    preproDf['Average Duration'] = AvgDuration
    preproDf['Total Duration'] = TotDur
    preproDf['Total Calls'] = TotCall
    preproDf['STD Incoming'] = STDTYPE['Incoming']
    preproDf['STD Outgoing'] = STDTYPE['Outgoing']
    preproDf['Unique Days Count'] = UniqueDayCnt
    preproDf['WeekDay Count'] = weekDayCnt
    preproDf['SaturDay Count'] = saturDayCnt
    preproDf['SunDay Count'] = sunDayCnt
    preproDf['Sat Call Time Per'] = SatCallPer
    preproDf['Sun Call Time Per'] = SunCallPer
    preproDf['Week Call Time Per'] = WeekCallPer
    preproDf['Calls Before Noon'] = CallsBfrNoon

    for i in ids:
        callLens = list(df.loc[df['Id'] == i][' Duration'])
        longContact.append(len(list(filter(lambda x: x >= int(
            2*preproDf.loc[preproDf['Id'] == i]['Average Duration']), callLens)))*100/int(preproDf.loc[preproDf['Id'] == i]['Frequency']))
        longTotal.append(len(list(filter(lambda x: x >= 2*TotalAvgDuration, callLens)))
                         * 100/int(preproDf.loc[preproDf['Id'] == i]['Frequency']))
    preproDf['% Long Total Calls'] = longTotal
    preproDf['% Long Contact Calls'] = longContact

    for i in ids:
        UserAvgDur = 0
        TotAvgDur = 0
        temp = df.loc[df['Id'] == i]
        tempdates = list(map(lambda x: x.date(), list(
            df.loc[df['Id'] == i][' Date_Time'])))
        weekdays = list(filter(lambda x: x.weekday() !=
                               5 and x.weekday() != 6, tempdates))
        x = preproDf.loc[preproDf['Id'] == i]['Average Duration']
        for idx, date in enumerate(tempdates):
            if date in weekdays:
                if(int(temp.iloc[idx, :][' Duration']) >= int(2*x)):
                    UserAvgDur = UserAvgDur+1
                if(temp.iloc[idx, :][' Duration'] >= 2*TotalAvgDuration):
                    TotAvgDur = TotAvgDur+1
        longWeekDayCallsTotal.append(
            TotAvgDur*100/int(preproDf.loc[preproDf['Id'] == i]['Frequency']))
        longWeekDayCallsContact.append(
            UserAvgDur*100/int(preproDf.loc[preproDf['Id'] == i]['Frequency']))
    preproDf['% Long Weekday Total Calls'] = longWeekDayCallsTotal
    preproDf['% Long Weekday Contact Calls'] = longWeekDayCallsContact
    #preproDf['Relationship'] = label
    return preproDf


def preproSmsData(filename):
    df = pd.read_csv(filename)
    #fn = filename.split('/')
    #username = fn[1].split('_')
    df['Id'] = df['ID']
    df[' Date_Time'] = df[' Date_Time'].apply(
        lambda x: datetime.datetime.fromtimestamp(int(x/1000)))
    for index, row in df.iterrows():
        names[row['Id']] = row[' Name']
    df[' Name'] = df[' Name'].apply(lambda x: "null")
    #df[' Relationship'] = df[' Relationship'].apply(lambda x: relationBMS[x])
    Mar = datetime.datetime(2020, 3, 1, 0, 0)
    postMar = df.loc[df[' Date_Time'] >= Mar]
    preMar = df.loc[df[' Date_Time'] < Mar]
    if len(postMar) > 0:
        postMarpreProDf = smsFunc2(postMar)
    else:
        postMarpreProDf = None
    if len(preMar) > 0:
        preMarpreProDf = smsFunc2(preMar)
    else:
        preMarpreProDf = None
    return postMarpreProDf, preMarpreProDf


def smsFunc2(df):
    ids = list(df.Id.unique())
    preproDf = pd.DataFrame(data={'Id': ids})
    freq = []
    UniqueDayCnt = []
    weekDayCnt = []
    saturDayCnt = []
    sunDayCnt = []
    SMSBfrNoon = []
    label = []
    for i in ids:
        temp = df.loc[df['Id'] == i]
        #rel = list(temp[' Relationship'])
        # label.append(rel[0])
        freq.append(len(temp))
        tempdates = list(map(lambda x: x.date(), list(
            df.loc[df['Id'] == i][' Date_Time'])))
        totalNoSMS = len(tempdates)
        weekdays = list(filter(lambda x: x.weekday() !=
                               5 and x.weekday() != 6, tempdates))
        weekDayCnt.append(len(weekdays)*100/totalNoSMS)
        saturdays = list(filter(lambda x: x.weekday() == 5, tempdates))
        saturDayCnt.append(len(saturdays)*100/totalNoSMS)
        sundays = list(filter(lambda x: x.weekday() == 6, tempdates))
        sunDayCnt.append(len(sundays)*100/totalNoSMS)
        uniquedays = len(set(tempdates))
        UniqueDayCnt.append(uniquedays)
        beforenoon = 0
        for time in list(temp[' Date_Time']):
            if time.hour < 12:
                beforenoon = beforenoon+1
        SMSBfrNoon.append(beforenoon*100/freq[-1])
    preproDf['Total_SMS'] = freq
    preproDf['Unique_Days_Count_SMS'] = UniqueDayCnt
    preproDf['WeekDay_Count_SMS'] = weekDayCnt
    preproDf['SaturDay_Count_SMS'] = saturDayCnt
    preproDf['SunDay_Count_SMS'] = sunDayCnt
    preproDf['SMS_Before_Noon'] = SMSBfrNoon
    #preproDf['Relationship'] = label
    return preproDf


def preprocessing(callCsv, smsCsv):
    postMarCall, preMarCall = preproCallData(callCsv)
    postMarSms, preMarSms = preproSmsData(smsCsv)
    preMrDf = pd.merge(preMarCall, preMarSms, on='Id', how='outer')
    postMrDf = pd.merge(postMarCall, postMarSms, on='Id', how='outer')
    preMrDf = preMrDf.fillna(method='ffill')
    preMrDf = preMrDf.fillna(0)
    preMrDf['Total Call and SMS'] = preMrDf['Total Calls'] + \
        preMrDf['Total_SMS']
    # preMrDfDf.to_csv("Pre_Mar_OuterJoin.csv")
    postMrDf = postMrDf.fillna(0)
    postMrDf['Total Call and SMS'] = postMrDf['Total Calls'] + \
        postMrDf['Total_SMS']
    # postMarDf.to_csv("Pre_Mar_OuterJoin.csv")
    model = pickle.load(open('randomForestFF.sav', 'rb'))
    preMarIds = list(preMrDf['Id'])
    postMarIds = list(postMrDf['Id'])

    # print(preMrDf)
    #preMrDf = preMrDf.drop(columns=['Unnamed: 0', 'Frequency', 'Unnamed: 0_y', 'Unnamed: 0_x', 'Id'])
    #postMrDf = postMrDf.drop(columns=['Unnamed: 0', 'Frequency', 'Unnamed: 0_y', 'Unnamed: 0_x', 'Id'])
    preMrDf = preMrDf.drop(columns=['Frequency', 'Id'])
    postMrDf = postMrDf.drop(columns=['Frequency', 'Id'])
    print(type(preMarIds[0]))
    X_preMar = (preMrDf[:]).to_numpy()
    X_postMar = (postMrDf[:]).to_numpy()
    y_pred_preMar = model.predict(X_preMar)
    y_pred_postMar = model.predict(X_postMar)
    y_preMar = [mapping[i] for i in y_pred_preMar]
    y_postMar = [mapping[i] for i in y_pred_postMar]
    print(len(postMarIds), len(y_postMar), len(names))
    resPreMar = {}
    resPostMar = {}
    print('For PreMarData')
    for inx, i in enumerate(preMarIds):
        resPreMar[names[i]] = y_preMar[inx]
    print('For PostMarData')
    for inx, i in enumerate(postMarIds):
        resPostMar[names[i]] = y_postMar[inx]
    json_preMar = json.dumps(resPreMar)
    json_postMar = json.dumps(resPostMar)
    print(json_preMar, json_postMar)
    return json_postMar
    #print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    #print(pd.crosstab(y_test, y_pred, rownames=['Actual Species'], colnames=['Predicted Species']))


# # main runner - moved to main.py
# preprocessing('sample_calls.csv', 'sample_SMS.csv')
