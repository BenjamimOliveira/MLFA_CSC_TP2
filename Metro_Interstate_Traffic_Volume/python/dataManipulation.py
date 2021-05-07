import pandas as pd

def getData():
    # dtypes for csv fields
    dtypes = {
        'holiday':str,
        'temp':float,
        'rain_1h':float,
        'snow_1h':float,
        'clouds_all':int,
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
    # itera pelas horas que não tem registo
    for x in range(registo1[4].hour+1,registo2[4].hour):
        print(x)
    print(registo1)
    print(registo2)

def preencherGaps(data):
    # iterar pelo dataset
    # pegar num registo, comparar hora com a hora do registo seguinte
    # se houver gap criar os registos
    data1 = data.to_numpy()
    print(data1[0])
    return "Wowie"


### --- MAIN SCRIPT --- ###
def __main__(): 
    # ler dados   
    data = getData()

    # Converter holidays para 0/1
    data = isHoliday(data)

    # preencher gaps
    data = preencherGaps(data)

    # 1 - preencher os gaps
    # 2 - remover repetidos
    # 3 - holiday -> 0/1
    # 4 - weather -> One hot encoding/order(?)
    # 5 - weekday vs weekend -> 0/1


data = getData()
#print(type(data.date_time.dt))
data = isHoliday(data)
data = data.loc[data['holiday']==1]
print(data)
#data1 = []
#data1.append(data[0])
#data1.append(data[3])
#gapF(data1[0], data1[1])