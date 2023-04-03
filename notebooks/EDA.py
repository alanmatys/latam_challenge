#!/usr/bin/env python
# coding: utf-8

# # EDA Latam Challenge!
# 
# ![Alt text](../docs/png-transparent-orlando-international-airport-latam-airlines-group-latam-chile-latam-brasil-logo-snapchat-purple-blue-violet.png)
# 
# - Author: Alan Matys
# - Linkedin: https://www.linkedin.com/in/alanmatys/

# In[391]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.distance import distance as geodesic_distance
import numpy as np
import os


# In[392]:


def read_df():
    """
    This function is reads the datataset_SCL.csv file and checks for any null 

    Parameters
    ---------------------------
    None

    Returns
    --------------------------

    df:pandas.DataFrame
        Dataframe containing the dataset corresponds to a flight that landed or took off from SCL during 2017.
    
    """
    
    df = pd.read_csv(os.path.join('..','data','raw','dataset_SCL.csv'),
        parse_dates=['Fecha-I','Fecha-O'])

    
    # NaN check in data frame
    if df.isnull().any().any():

        print('There is/are NaN in these Rows\n',df.isnull().any())
    
    if df.duplicated().sum():

        print('There is/are duplicanted',df.duplicated().sum(),' rows')

    return df


# In[393]:


df = read_df()
df.head()


# In[394]:


def synthetic_features(df):
    """
    This function's objective is to create the synthetic features required on the Challenge, including:

    
    - high_season: 1 if Date-I is between Dec-15 and Mar-3, or Jul-15 and Jul-31, or Sep-11 and Sep-30, 0 otherwise.

    - min_diff: difference in minutes between Date-O and Date-I .

    - delay_15: 1 if min_diff > 15, 0 if not.

    - period_day:morning (between 5:00 and 11:59), afternoon (between 12:00 and 18:59) and night (between 19:00 and 4:59), based on Date-I.

    If the 'synthetic_features.csv' file already exists in data/interim, it concatenates both dfs 

    Parameters
    -------------------------
    df: pandas.DataFrame
        DataFrame corresponding to the Latam Challenge

    Returns
    -------------------------
    df: pandas.DataFrame
        Dataframe with the syntetic features added as columns
    
    """

    # Check if the synthetic_features file exists, concat and return concated df
    if os.path.exists(os.path.join('..','data','interim','synthetic_features.csv')):

        df_synt = pd.read_csv(os.path.join('..','data','interim','synthetic_features.csv'))

        df = pd.concat([df,df_synt], axis=1)
        
        return df

    # New columns to create
    new_cols = ['high_season','min_diff','delay_15','period_day']

    # Define high_season
    high_season = (
                (df['Fecha-I'].between('2016-12-15','2017-03-03')) 
                | (df['Fecha-I'].between('2017-09-11','2017-09-30')) 
                | (df['Fecha-I'].between('2017-07-15','2017-07-31'))
    )

    df[new_cols[0]] = np.where(high_season, 
                               1,
                               0)

    df[new_cols[1]] = (df['Fecha-O'] - df['Fecha-I']).dt.total_seconds() / 60.0

    # Create the 
    df[new_cols[2]] = np.where(df['min_diff'] > 15,
                              1,
                              0)
    
    # Create a function to classify the time of the day
    def classify_time(time):
        if 5 <= time.hour <= 11:
            return 'morning'
        elif 12 <= time.hour <= 18:
            return 'afternoon'
        else:
            return 'night'

    # Create the period_day column with the previously defined funciton
    df[new_cols[3]] = df['Fecha-I'].apply(classify_time)

    
    df.loc[:,new_cols].to_csv(os.path.join('..','data','interim','synthetic_features.csv'), index=False)

    return df


# In[395]:


df = synthetic_features(df)
df.head()


# ## EDA (Exploratory Data Analysis)
# 
# In this section we try to describe the categorical and the numerical variables

# ### Categorical Features

# In[396]:


def programed_vs_operated(df:pd.DataFrame):
    """

    Plot the comparision of the Programed features vs the Operated features

    Parameters
    --------------------

    - df: pandas.DataFrame
        Pandas DataFrame containing the this columns:
        - Origin Variables (Ori-I/Ori-O)
        - Destination Variables (Des-I/Des-O)
    
    """
    fig, axes = plt.subplots(nrows=3,ncols=2,figsize=(20,12))

    fig.suptitle('Programmed vs Operation')

    sns.countplot(x=df["Ori-I"],
                order = df['Ori-I'].value_counts().index,
                ax=axes[0,0],
                color='red',
                alpha=0.7)

    sns.countplot(x=df["Ori-O"],
                order = df['Ori-O'].value_counts().index,
                ax=axes[0,1],
                color='blue',
                alpha=0.7)

    sns.countplot(x=df["Des-I"],
                order = df['Des-I'].value_counts().index,
                ax=axes[1,0],
                color='red',
                alpha=0.7)
    axes[1,0].tick_params(axis='x', rotation=90)

    sns.countplot(x=df["Des-O"],
                order = df['Des-O'].value_counts().index,
                ax=axes[1,1],
                color='blue',
                alpha=0.7)
    axes[1,1].tick_params(axis='x', rotation=90)

    dis_des = round((len(df[df['Des-I'] != df['Des-O']])/len(df)) * 100,2)

    axes[1,1].text(.4, .70,
                    f'{dis_des}% of the flights that ended on a different destination',
                    ha='left',
                    va='top',
                    color='blue',
                    transform=axes[1,1].transAxes)

    sns.countplot(x=df["Emp-I"],
                order = df['Emp-I'].value_counts().index,
                ax=axes[2,0],
                color='red',
                alpha=0.7)
    axes[2,0].tick_params(axis='x', rotation=90)

    sns.countplot(x=df["Emp-O"],
                order = df['Emp-O'].value_counts().index,
                ax=axes[2,1],
                color='blue',
                alpha=0.7)
    axes[2,1].tick_params(axis='x', rotation=90)

    dis_emp = round((len(df[df['Emp-I'] != df['Emp-O']])/len(df)) * 100,2)

    axes[2,1].text(.4, .70,
                    f'{dis_emp}% of the flights that have a diifferent Airline Code',
                    ha='left',
                    va='top',
                    color='blue',
                    transform=axes[2,1].transAxes)

    plt.show()


# In[397]:


programed_vs_operated(df)


# From this chart we can take the following insights
# - More than 1/4 flights had a different airline code from the one that was programed.
# - There are some flights, not usually, that ended on a different destination, based on the destination code

# In[398]:


def other_categorical(df):
    """

    Plot other categorical Variables of the LATAM Dataframe

    Parameters
    --------------------
    df: pandas.Dataframe
        Latam Dataframe containing the 'TIPOVUELO', 'DIANOM', 'SIGLADES', 'OPERA' columns

    Returns
    -------------------
    None
    
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (20,6))

    sns.histplot(x=df["TIPOVUELO"],
                    multiple="dodge", 
                    stat = 'percent',
                    shrink = 0.8,
                    common_norm=False,
                    ax = axes[0,0])

    sns.histplot(x=df["DIANOM"],
                    multiple="dodge", 
                    stat = 'percent',
                    shrink = 0.8,
                    ax = axes[0,1],
                    discrete=True,
                    hue = df['TIPOVUELO'])

    axes[0,1].get_legend().remove()

    sns.countplot(x=df['SIGLADES'],
                ax = axes[1,0],
                order = df['SIGLADES'].value_counts().index,
                hue= df['TIPOVUELO']
                )

    axes[1,0].tick_params(axis='x', rotation=90)
    axes[1,0].set(xlabel=None)

    sns.histplot(x=df["OPERA"],
                    multiple="dodge", 
                    stat = 'percent',
                    shrink = 0.8,
                    ax = axes[1,1],
                    discrete=True,
                    hue = df['TIPOVUELO'])

    axes[1,1].tick_params(axis='x', rotation=90)
    axes[1,1].set(xlabel=None)


    plt.show()


# In[399]:


other_categorical(df)


# From this chart we can take the following insights:
# - FROM Chart 1:
# 
# There are more National Flights (within Chile) than International Flights (from SCL to another country), but more or less is equivalent.
# 
# - From Chart 2:
# 
# International flights are uniform in terms of the day of the flight, but for National flights it's not the same, there are significantly fewer flights on Saturdays
# 
# - From Chart 3:
# 
# The top destination is Buenos Aires, the second place is for Antofagasta. On the top 10 flights most of them are national flights.
# 
# - From Chart 4:
# 
# Latam clearly dominates both National and International Flights in SCL. On second place we have Sky Airline with a significant market on National and some on International. With Latam & Sky we have ~ 80% of the flights
# 
# The others seem far from Top 2 players
# 

# ### Consistency of results in time
# 
# We examine if the dominance from both airlines and destinations has persisted in time for the full year or from certain perioid and on they gained full dominance.

# In[400]:


def time_s_airlines(df, var):
    """
    
    """
    fig,ax = plt.subplots(nrows = 4,figsize=(20,28))

    plot = df.groupby([df['Fecha-I'].dt.date,df[var]]).agg(vuelos = ('Vlo-I','count'))

    sns.lineplot(plot,x='Fecha-I',y='vuelos',hue=var, ax= ax[0])

    ax[0].set_title('Flights per Day')
    ax[0].set(xlabel=None)

    sns.move_legend(
    ax[0], loc="center left",
    bbox_to_anchor=(1, .7), ncol=2, title=None
    )


    plot_2 = df[~df.OPERA.isin(['Grupo LATAM','Sky Airline'])].groupby([df['Fecha-I'].dt.date,df[var]]).agg(vuelos = ('Vlo-I','count'))

    sns.lineplot(plot_2,x='Fecha-I',y='vuelos',hue=var, ax= ax[1])


    sns.move_legend(
    ax[1], loc="center left",
    bbox_to_anchor=(1, .7), ncol=2, title=None
    )

    ax[1].set_title('Flights Excluding LATAM & Sky')
    ax[1].set(xlabel=None)


    df_nac = df[df['TIPOVUELO'] == 'N']

    plot_nac = df_nac.groupby([df_nac['Fecha-I'].dt.date,df[var]]).agg(vuelos = ('Vlo-I','count'))

    sns.lineplot(plot_nac,x='Fecha-I',y='vuelos',hue=var, ax=ax[2])

    sns.move_legend(
    ax[2], loc="center left",
    bbox_to_anchor=(1, .9), ncol=2, title=None, fancybox=True, shadow=True
    )


    ax[2].set_title('Nac Flights')
    ax[2].set(xlabel=None)


    df_int = df[df['TIPOVUELO'] == 'I']

    plot_int = df_int.groupby([df_int['Fecha-I'].dt.date,df[var]]).agg(vuelos = ('Vlo-I','count'))

    sns.lineplot(plot_int,x='Fecha-I',y='vuelos',hue=var)


    ax[3].set_title('Int Flights')
    ax[3].set(xlabel=None)

    sns.move_legend(
    ax[3], loc="center left",
    bbox_to_anchor=(1, .7), ncol=2, title=None
    )

    
    plt.show()


# In[401]:


time_s_airlines(df,'OPERA')


# In[403]:


time_s_airlines(df, 'SIGLADES')


# ## Continuous Features

# In[404]:


sns.displot(x=df['DIANOM'],bins=12,hue=df['TIPOVUELO'])


# In[405]:


sns.displot(x=df['MES'], bins=12,hue=df['SIGLADES'])


# In[406]:


sns.displot(x=df['DIA'], bins=31,hue=df['TIPOVUELO'])


# ## Extra Features
# 
# We will add some features from the dataset and exogenous sources.
# 

# In[414]:


def geo_data(df:pd.DataFrame):
    """
    Create a geo.csv file that has te distance between the origin city and the destination city

    Parameters
    ------------------------
    df: pandas.DataFrame
        Latam Challenge Dataframe containing the origin column and the destination column from each flight.
    
    """


    # Check if the geo_features file exists, concat and return concated df
    if os.path.exists(os.path.join('..','data','interim','geo_features.csv')):

        df_geo = pd.read_csv(os.path.join('..','data','interim','geo_features.csv'))

        df_out = df.merge(df_geo.loc[:,['Dest','Orig','distance','country_des']], left_on= ['SIGLADES','SIGLAORI'], right_on=['Dest','Orig'])
        
        return df_out

    df_out = pd.DataFrame()
    df_out['Dest'] = df['SIGLADES'].unique()
    df_out['Orig'] = df['SIGLAORI'].unique().repeat(len(df['SIGLADES'].unique()))

    # Define a function to geocode a city
    geolocator = Nominatim(user_agent="test_latam_airlines")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    
    def get_coordinates(city):
        location = geocode(city)
        if location is not None:
            return location.latitude, location.longitude
        else:
            return None, None
    
    
    def get_country(coord):
        location = geolocator.reverse(coord).raw['address']
        if location is not None:
            return location.get('country','')
        else:
            return None

    #len(df['SIGLADES'].unique())
    df_out['coord_ori'] = list(map(get_coordinates, df_out['Orig']))
    df_out['coord_des'] = list(map(get_coordinates, df_out['Dest']))
    df_out['country_des'] = list(map(get_country, df_out['coord_des']))

    # Add the latitude and longitude coordinates to the flight data
    df_out[['ori_lat','ori_long']] = pd.DataFrame(df_out['coord_ori'].tolist(), index=df_out.index)
    df_out[['des_lat','des_long']] = pd.DataFrame(df_out['coord_des'].tolist(), index=df_out.index)


    # Calculate distance between cities
    df_out["distance"] = df_out.apply(lambda row: geodesic_distance(row['coord_ori'], row['coord_des']).km, axis=1)

    # Save to geo_features
    df_out.to_csv(os.path.join('..','data','interim','geo_features.csv'), index=False)
    

    df.merge(df_out.loc[:,['Dest','Orig','distance','country_des']], left_on= ['SIGLADES','SIGLAORI'], right_on=['Dest','Orig'])

    return df


# In[415]:


df_geo = geo_data(df)
df_geo.head()

