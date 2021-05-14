import pandas as pd
import datetime
from operator import itemgetter
import functools

def getData():
    # dtypes for csv fields
    dtypes = {
        'holiday':str,
        'temp':float,
        'rain_1h':float,
        'snow_1h':float,
        'clouds_all':float,
        'weather_main':str,
        'weather_description':str,
        'date_time':str,
        'traffic_volume':int
        }
    # dates to be parsed from the csv
    parse_dates = ['date_time']

    # read csv
    data = pd.read_csv("../data/Metro_Interstate_Traffic_Volume.csv", dtype=dtypes, parse_dates=parse_dates)
    data['date_time'] = pd.to_datetime(data.date_time, format='%Y-%m-%d %H:%M:%S', errors='raise')

    # drop unwanted columns
    unwanted_cols = ['weather_description', 'rain_1h', 'snow_1h'] 
    data = data.drop(unwanted_cols, axis=1)

    # sort by date
    data = data.sort_values(by=['date_time'])
    return data

def isHoliday(data):
    data.loc[(data.holiday == 'None'), 'holiday']=0
    data.loc[(data.holiday != 0), 'holiday']=1
    dataH = data.loc[(data.holiday != 0)]

    # holidays são apenas registados na primeira hora de cada dia
    # propagar o holiday ás restantes horas do diax
    for index, row in dataH.iterrows():
        # row possui todas as datas de feriados do dataset (1 por dia) 
        data.loc[(data.date_time.dt.date == row.date_time.date()), 'holiday'] = 1
    return data

# A utilizar dentro de preencherGaps(data)
def gapF(registo1, registo2):
    
    holi = registo1.holiday
    temp = (registo1.temp + registo2.temp)/2
    clouds = (registo1.clouds_all + registo2.clouds_all)/2
    weather = registo1.weather_main
    traffic = (registo1.traffic_volume + registo2.traffic_volume)/2

    registosGen = []

    # se o gap for dentro de um só dia
    if registo1[4].hour+1 < registo2[4].hour or registo2[4].hour==0:
        hora2 = registo2[4].hour
        if registo2[4].hour==0:
            hora2 = 24
        # itera pelas horas que não tem registo
        for x in range(registo1[4].hour+1,hora2):
            dict = {
                'holiday':holi,
                'temp':temp,
                'clouds_all':clouds,
                'weather_main':weather,
                'date_time':datetime.datetime(registo1[4].year, registo1[4].month, registo1[4].day, x, 00, 00),
                'traffic_volume':traffic
            }

            dict = pd.Series(dict)
            registosGen.append(dict)
    # se o gap for de um dia para outro (22h de um dia para as 2h doutro, p/ ex)
    else:
        for x in range(registo1[4].hour+1,24):
            dict = {
                'holiday':holi,
                'temp':temp,
                'clouds_all':clouds,
                'weather_main':weather,
                'date_time':datetime.datetime(registo1[4].year, registo1[4].month, registo1[4].day, x, 00, 00),
                'traffic_volume':traffic
            }

            dict = pd.Series(dict)
            registosGen.append(dict)
        for x in range(0,registo2[4].hour):
            dict = {
                'holiday':holi,
                'temp':temp,
                'clouds_all':clouds,
                'weather_main':weather,
                'date_time':datetime.datetime(registo1[4].year, registo1[4].month, registo1[4].day+1, x, 00, 00),
                'traffic_volume':traffic
            }

            dict = pd.Series(dict)
            registosGen.append(dict)

    return registosGen

def preencherGaps(data):
    # iterar pelo dataset
    # pegar num registo, comparar hora com a hora do registo seguinte
    # se houver gap criar os registos
    data1 = data
    conc = []
    for x in range(len(data1)-1):
        # Introduzir aqui comparação entre hora atual e a próxima do registo
        if data1.iloc[x].date_time.hour+1 != data1.iloc[x+1].date_time.hour:
            #print(data1.iloc[x].date_time.hour)
            #print(data1.iloc[x+1].date_time.hour)
            try:
                conc = conc + gapF(data1.iloc[x], data1.iloc[x+1])
            except:
                print("linha ", x)
                #print(data1.iloc[x-1])
                print(data1.iloc[x])
                #print(data1.iloc[x+1])
            #print('-------')

    #print("******* concat *******")
    for x in conc:
        data = data.append(x,ignore_index=True)
    data = data.sort_values('date_time')
    return data

def encodeWeather(data):
    
    return data

# 0 - weekday | 1 - weekend
def encodeWeekends(data):
    weekday = [0]*len(data)
    data['weekend'] = weekday
    data.loc[(data.date_time.dt.weekday == 5), 'weekend'] = 1
    data.loc[(data.date_time.dt.weekday == 6), 'weekend'] = 1
    return data
### --- MAIN SCRIPT --- ###
def __main__(): 
    # ler dados   
    data = getData()

    # Converter holidays para 0/1
    data = isHoliday(data)

    # Remover repetidos
    data = data.drop_duplicates(subset='date_time')

    # preencher gaps
    data = preencherGaps(data)

    # Encode weather
    data = encodeWeather(data)

    # Encode Weekends
    data = encodeWeekends(data)
    
    data['clouds_all'] = data['clouds_all'].astype(int)
    data['traffic_volume'] = data['traffic_volume'].astype(int)
    data.to_csv('../data/metro_processed.csv', index=False)

__main__()
