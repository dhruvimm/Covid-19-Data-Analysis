import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.figure_factory as ff 
import plotly.express as px
import folium
import plotly.graph_objects as go

def scrape_world ():
    url="https://www.worldometers.info/coronavirus/"
    headers={'User-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36'}
    response=requests.get(url, headers=headers)
    soup= BeautifulSoup(response.content,'html.parser')
    coronatable = soup.find_all('table')
    ct = coronatable[0]
    country = []
    total_cases = []
    new_cases = []
    total_deaths = []
    new_deaths = []
    total_recovered = []
    active_cases = []
    serious_critical=[]
    total_cases_1m=[]
    deaths_1m=[]
    total_tests=[]
    test_1m=[]
    population=[]
    rows = ct.find_all('tr')[9:-8]
    for row in rows:
        col = row.find_all('td')
        country.append(col[1].text.strip())
        total_cases.append(col[2].text.strip().replace(',',''))
        new_cases.append(col[3].text.strip().replace(',','').replace('+',''))
        total_deaths.append(col[4].text.strip().replace(',',''))
        new_deaths.append(col[5].text.strip().replace(',','').replace('+',''))
        total_recovered.append(col[6].text.strip().replace(',',''))
        active_cases.append(col[8].text.strip().replace(',','').replace('+',''))
        serious_critical.append(col[9].text.strip().replace(',',''))
        total_cases_1m.append(col[10].text.strip().replace(',',''))
        deaths_1m.append(col[11].text.strip().replace(',',''))
        total_tests.append(col[12].text.strip().replace(',',''))
        test_1m.append(col[13].text.strip().replace(',',''))
        population.append(col[14].text.strip().replace(',',''))

    world_df = pd.DataFrame(list(zip(country,new_cases, active_cases, total_recovered, new_deaths, total_deaths, total_cases,serious_critical,total_cases_1m,deaths_1m,total_tests,test_1m,population)),
                  columns = ['Country','NewCases','ActiveCases','TotalRecovered','NewDeaths','TotalDeaths','TotalCases','Serious_or_Critical','Cases_per_1M_Pop','Deaths_per_1M_Pop','TotalTests','Tests_per_1M_Pop','Population'])


    # Cleaning the Data
    # Handling missing values by replacing them with NaN

    world_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    world_df.replace('N/A', np.nan, inplace=True)

    # Missing values in NewCases and New Deaths can be filled with zeros 

    world_df.NewCases.replace(np.nan,0,inplace=True)
    world_df.NewDeaths.replace(np.nan,0,inplace=True)

    # For ActiveCase, TotalRecovered and TotalDeaths 
    # The missing values can be replaced using the expression : 
    # TotalCases = ActiveCases + TotalRecovered + TotalDeaths

    for i in world_df.index:
        if world_df.ActiveCases[i] is np.nan:
            if world_df.TotalRecovered[i] is np.nan or world_df.TotalDeaths[i] is np.nan:
                world_df.ActiveCases[i] = 0 + int(world_df.NewCases[i])
            else:
                world_df.ActiveCases[i] = int(world_df.TotalCases[i]) + int(world_df.NewCases[i]) - int(world_df.TotalDeaths[i]) - int(world_df.TotalRecovered[i])
        if world_df.TotalRecovered[i] is np.nan:
            if world_df.TotalDeaths[i] is np.nan:
                world_df.TotalRecovered[i] = 0
            else:
                world_df.TotalRecovered[i] = int(world_df.TotalCases[i]) - int(world_df.TotalDeaths[i]) - int(world_df.ActiveCases[i])
                
        if world_df.TotalDeaths[i] is np.nan:
            world_df.TotalDeaths[i] = int(world_df.TotalCases[i]) + int(world_df.NewDeaths[i]) - int(world_df.ActiveCases[i]) - int(world_df.TotalRecovered[i])

    world_df['Serious_or_Critical'] = world_df['Serious_or_Critical'].fillna(0)

    world_df.NewCases = world_df.NewCases.astype(np.int64)
    world_df.ActiveCases = world_df.ActiveCases.astype(np.int64)
    world_df.TotalRecovered = world_df.TotalRecovered.astype(np.int64)
    world_df.NewDeaths = world_df.NewDeaths.astype(np.int64)
    world_df.TotalDeaths = world_df.TotalDeaths.astype(np.int64)
    world_df.TotalCases = world_df.TotalCases.astype(np.int64)
    world_df.Serious_or_Critical = world_df.Serious_or_Critical.astype(np.int64)
    world_df.Tests_per_1M_Pop = world_df.Tests_per_1M_Pop.fillna(0).astype(np.int64)
    world_df.Deaths_per_1M_Pop = world_df.Deaths_per_1M_Pop.fillna(0).astype(float)
    world_df.Deaths_per_1M_Pop = world_df.Deaths_per_1M_Pop.astype(np.int64)
    world_df.Cases_per_1M_Pop = world_df.Cases_per_1M_Pop.fillna(0).astype(np.int64)
    world_df.TotalTests = world_df.TotalTests.fillna(0).astype(np.int64)

    world_df = world_df.drop(columns=['NewDeaths', 'NewCases'])
    return world_df
    
def scrape_india():
    india_url = 'https://www.mohfw.gov.in/'
    response = requests.get(india_url)
    soup = BeautifulSoup(response.content,'html.parser')
    ct = soup.find('table')
    state = []
    total_cases = []
    deaths = []
    recovered = []
    active_cases = []
    rows = ct.find_all('tr')[1:35]
    for row in rows:
        col = row.find_all('td')
        state.append(col[1].text.strip())
        active_cases.append(col[2].text.strip())
        recovered.append(col[3].text.strip())
        deaths.append(col[4].text.strip())
        total_cases.append(col[5].text.strip())
        
    india_df = pd.DataFrame(list(zip(state,active_cases,recovered,deaths,total_cases)),
                            columns=['State','Active','Recovered','Deaths','TotalCases'])

    india_df.Active = india_df.Active.astype(int)
    india_df.Recovered = india_df.Recovered.astype(int)
    india_df.Deaths = india_df.Deaths.astype(int)
    india_df.TotalCases = india_df.TotalCases.astype(int)


    return india_df

def table_india():
    df=scrape_india()
    
    df3 = ff.create_table(df)
    # return df3
    return plotly.offline.plot(df3,output_type='div')  

def table_world():
    df=scrape_world()
    df=df.copy().drop(columns=['Serious_or_Critical','Cases_per_1M_Pop','Population'])
    df3 = ff.create_table(df)
    # return df3
    return plotly.offline.plot(df3,output_type='div')  
def total_world():
    df=scrape_world()
    Total_Cases=format(df["TotalCases"].sum(),',d')
    Active_Cases=format(df["ActiveCases"].sum(),',d')
    Total_Deaths=format(df["TotalDeaths"].sum(),',d')
    Recovered_Cases=format(df["TotalRecovered"].sum(),',d')
    return [Total_Cases,Active_Cases,Recovered_Cases,Total_Deaths]

def total_india():
    df=scrape_india()
    Total_Cases=format(df["TotalCases"].sum(),',d')
    Active_Cases=format(df["Active"].sum(),',d')
    Total_Deaths=format(df["Deaths"].sum(),',d')
    Recovered_Cases=format(df["Recovered"].sum(),',d')
    return [Total_Cases,Active_Cases,Recovered_Cases,Total_Deaths]

def plot_world():
    df=scrape_world()
    df=df.head(30)
    df = df.sort_values('TotalCases', ascending=False).fillna(0)
    fig= go.Figure(data=[
        go.Bar(name='TotalCases', x=df['Country'],y=df['TotalCases']),
        go.Bar(name='Recovered', x=df['Country'],y=df['TotalRecovered']),
        go.Bar(name='Active', x=df['Country'],y=df['ActiveCases']),
        go.Bar(name='Deaths', x=df['Country'],y=df['TotalDeaths'])
    ])
    fig.update_layout(barmode='overlay')
    return plotly.offline.plot(fig,output_type='div')

def plot_india():
    df=scrape_india()
    df=df.head(30)
    df = df.sort_values('TotalCases', ascending=False).fillna(0)
    fig= go.Figure(data=[
        go.Bar(name='TotalCases', x=df['State'],y=df['TotalCases']),
        go.Bar(name='Recovered', x=df['State'],y=df['Recovered']),
        go.Bar(name='Active', x=df['State'],y=df['Active']),
        go.Bar(name='Deaths', x=df['State'],y=df['Deaths'])
    ])
    fig.update_layout(barmode='overlay')
    return plotly.offline.plot(fig,output_type='div')

def plot_tests():
    df=scrape_world()
    df=df.head(30)
    df = df.sort_values('Tests_per_1M_Pop', ascending=False).fillna(0)
    fig = px.bar(df, x='Country', y='Tests_per_1M_Pop', color='Tests_per_1M_Pop', height=600)
    return plotly.offline.plot(fig,output_type='div')

def plot_severity():
    df=scrape_world()
    df=df.head(30)
    df=df.sort_values('Tests_per_1M_Pop', ascending=False)
    x= df['Deaths_per_1M_Pop']
    y= df['Cases_per_1M_Pop']
    jittered_y = y + (y*.1) * np.random.rand(len(y)) -0.05
    jittered_x = x + (x*.1) * np.random.rand(len(x)) -0.05
    df['Deaths_per_1M']=jittered_x
    df['Confirmed_per_1M']=jittered_y
    r_to_d = df['TotalRecovered']/df['TotalDeaths']
    df['R_to_D']=r_to_d
    fig=px.scatter(df,x='Deaths_per_1M',y='Confirmed_per_1M',color='Tests_per_1M_Pop',size='R_to_D',
                hover_data=['Deaths_per_1M','Confirmed_per_1M','Country','Tests_per_1M_Pop'],text='Country')
    fig.update_traces(textposition='top center')
    fig.update_layout(yaxis_type='log',xaxis_title="Deaths/Million",yaxis_title="Confirmed cases / Million (log scale)")
    return plotly.offline.plot(fig,output_type='div')

def plot_daily():
    covid19India = pd.read_csv('covid_19_india.csv')
    covid19India['Date'] = pd.to_datetime(covid19India['Date'],dayfirst=True)
    df=covid19India.groupby('Date').sum()
    df.reset_index(inplace=True)
    fig= go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Confirmed'],mode='lines+markers',name='Confirmed'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Cured'],mode='lines+markers',name='Cured'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Deaths'],mode='lines+markers',name='Deaths'))
    return plotly.offline.plot(fig,output_type='div')

def plot_recovery():
    df=scrape_india()
    df["RecoveryRate"] = np.round(100*df["Recovered"]/df["TotalCases"],2)
    df=df.sort_values("RecoveryRate", ascending=False)
    fig = px.bar(df, x='State', y='RecoveryRate', color='RecoveryRate', height=600)
    return plotly.offline.plot(fig,output_type='div')  

def plot_age():
    ageGroup = pd.read_csv('AgeGroupDetails.csv')
    fig = px.bar(ageGroup, x='AgeGroup', y='TotalCases', color='TotalCases',text='Percentage', height=600)
    return plotly.offline.plot(fig,output_type='div')

def summary_world():
    df=scrape_world()
    world_summary=[]
    world_summary.append(df['TotalCases'].sum())
    world_summary.append(df['ActiveCases'].sum())
    world_summary.append(df['TotalRecovered'].sum())
    world_summary.append(df['TotalDeaths'].sum())

    fig = go.Figure(data=[go.Pie(labels=['Active','Recovered','Deaths'], 
                                values=[world_summary[1],world_summary[2],world_summary[3]], 
                                textinfo='label+percent',textposition='inside')])
    fig.update_layout(showlegend=False,
                    title ={'text' : "Confirmed Cases in the World : "+ format(world_summary[0],',d'),
                     'y':0.9, 'x':0.5, 'xanchor': 'center','yanchor': 'top'}) 
    return plotly.offline.plot(fig,output_type='div')

def summary_india():
    df=scrape_india()
    india_summary=[]
    india_summary.append(df['TotalCases'].sum())
    india_summary.append(df['Active'].sum())
    india_summary.append(df['Recovered'].sum())
    india_summary.append(df['Deaths'].sum())

    fig = go.Figure(data=[go.Pie(labels=['Active','Recovered','Deaths'], 
                                values=[india_summary[1],india_summary[2],india_summary[3]], 
                                textinfo='label+percent',textposition='inside')])
    fig.update_layout(showlegend=False,
                    title ={'text' : "Confirmed Cases in India : "+ format(india_summary[0],',d'),
                     'y':0.9, 'x':0.5, 'xanchor': 'center','yanchor': 'top'})
    return plotly.offline.plot(fig,output_type='div')