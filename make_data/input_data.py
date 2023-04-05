import os
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.distance import distance as geodesic_distance
import requests
import holidays


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

    df = pd.read_csv(
        os.path.join("..", "data", "raw", "dataset_SCL.csv"),
        parse_dates=["Fecha-I", "Fecha-O"],
    )

    # NaN check in data frame
    if df.isnull().any().any():
        print("There is/are NaN in these Rows\n", df.isnull().any())

        print('The row that has the NaN is:\n', df[df.isnull().any(axis=1)])

    if df.duplicated().sum():
        print("There is/are duplicanted", df.duplicated().sum(), " rows")
    
    else:
        print("\nThere are no duplicate rows")

    return df




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



def geo_data(df:pd.DataFrame):
    """
    Create a geo.csv file that has te distance between the origin city and the destination city

    Parameters
    ------------------------
    df: pandas.DataFrame
        Latam Challenge Dataframe containing the origin column and the destination column from each flight.

    Returns
    -----------------------
    None
    
    - Saves geo_data as csv into the data/interim folder

    """
    # Correct the coordinates for the following destinations that recive the wrong ones
    def correct_cord(row):
        if row['Dest'] == 'Concepcion':
            return (-36.82699, -73.04977)
            
        elif row['Dest'] == 'Osorno':
            return (-40.57395, -73.13348)
        
        elif row['Dest'] == 'Ushuia':
            return (-54.8, -68.3)

        elif row['Dest'] == 'Curitiba, Bra.':
            return (-25.441105, -49.276855)
        
        elif row['Dest'] == 'San Juan, Arg.':
            return (-31.5375, -68.53639)

        else:  
            return row['coord_des']

    # Check if the geo_features file exists, concat and return concated df
    if os.path.exists(os.path.join('..','data','interim','geo_features.csv')):

        df_geo = pd.read_csv(os.path.join('..','data','interim','geo_features.csv'))

        # Apply the correction to the coordinates column
        df_geo["coord_des"] = df_geo.apply(correct_cord, axis=1)

        # Correct the country for the Concepcion & Osorno destination that had an error
        df_geo['country_des'] = np.where((df_geo['Dest'].isin(['Concepcion','Osorno'])),'Chile', df_geo['country_des'])
        df_geo['country_des'] = np.where((df_geo['Dest'].isin(['Ushuia','San Juan, Arg.'])),'Argentina', df_geo['country_des'])
        df_geo['country_des'] = np.where((df_geo['Dest'].isin(['Curitiba, Bra.'])),'Brasil', df_geo['country_des'])

        df_out = df.merge(df_geo.loc[:,['Dest','Orig','distance','country_des','des_long','des_lat']], 
                          left_on= ['SIGLADES','SIGLAORI'], 
                          right_on=['Dest','Orig'])

        
        # get the holidays in chile
        cl_holidays = holidays.Chile()

        # Apply to define a holiday
        df_out['holiday'] = df_out['Fecha-I'].apply(lambda x: x in cl_holidays).astype(int)
        
        return df_out

    # Create a dataframe, extract the unique destination and origin for the entire dataframe 
    df_out = pd.DataFrame()
    df_out['Dest'] = df['SIGLADES'].unique()
    df_out['Orig'] = df['SIGLAORI'].unique().repeat(len(df['SIGLADES'].unique()))

    # Define a function to geocode a city
    geolocator = Nominatim(user_agent="test_latam_airlines")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    
    # Extract Longitude & Latitude from get_coordinates function using city name from geopy
    def get_coordinates(city):
        location = geocode(city)
        if location is not None:
            return location.latitude, location.longitude
        else:
            return None, None
    
    # Extract Country name from get_country function using coordinates (tuple) from geopy
    def get_country(coord):
        location = geolocator.reverse(coord).raw['address']
        if location is not None:
            return location.get('country','')
        else:
            return None

    # Get the coordinates for the destination, the origin and get the country of destination
    df_out['coord_ori'] = list(map(get_coordinates, df_out['Orig']))
    df_out['coord_des'] = list(map(get_coordinates, df_out['Dest']))
    df_out['country_des'] = list(map(get_country, df_out['coord_des']))
    

    # Correct the coordinates for the following destinations that recive the wrong ones
    def correct_cord(row):
        if row['Dest'] == 'Concepcion':
            return (-36.82699, -73.04977)
            
        elif row['Dest'] == 'Osorno':
            return (-40.57395, -73.13348)
        
        elif row['Dest'] == 'Ushuia':
            return (-54.8, -68.3)

        elif row['Dest'] == 'Curitiba, Bra.':
            return (-25.441105, -49.276855)
        
        elif row['Dest'] == 'San Juan, Arg.':
            return (-31.5375, -68.53639)

        else:  
            return row['coord_des']
        
    # Apply the correction to the coordinates column
    df_out["coord_des"] = df_out.apply(correct_cord, axis=1)

    # Correct the country for the Concepcion & Osorno destination that had an error
    df_out['country_des'] = np.where((df_out['Dest'].isin(['Concepcion','Osorno'])),'Chile', df_out['country_des'])
    df_out['country_des'] = np.where((df_out['Dest'].isin(['Ushuia','San Juan, Arg.'])),'Argentina', df_out['country_des'])
    df_out['country_des'] = np.where((df_out['Dest'].isin(['Curitiba, Bra.'])),'Brasil', df_out['country_des'])
    

    # Add the latitude and longitude coordinates to the flight data
    df_out[['ori_lat','ori_long']] = pd.DataFrame(df_out['coord_ori'].tolist(), index=df_out.index)
    df_out[['des_lat','des_long']] = pd.DataFrame(df_out['coord_des'].tolist(), index=df_out.index)


    # Calculate distance between cities
    df_out["distance"] = df_out.apply(lambda row: geodesic_distance(row['coord_ori'], row['coord_des']).km, axis=1)

    # Save to geo_features
    df_out.to_csv(os.path.join('..','data','interim','geo_features.csv'), index=False)
    
    # merge to the input df
    df = df.merge(df_out.loc[:,['Dest','Orig','distance','country_des','des_long','des_lat']], left_on= ['SIGLADES','SIGLAORI'], right_on=['Dest','Orig'])

    # get the holidays in chile
    cl_holidays = holidays.Chile()

    # Apply to define a holiday
    df['holiday'] = df['Fecha-I'].apply(lambda x: x in cl_holidays).astype(int)

    return df