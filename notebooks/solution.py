#!/usr/bin/env python
# coding: utf-8

# # EDA Latam Challenge!
#
# ![Alt text](../docs/png-transparent-orlando-international-airport-latam-airlines-group-latam-chile-latam-brasil-logo-snapchat-purple-blue-violet.png)
#
# - Author: Alan Matys
# - Linkedin: https://www.linkedin.com/in/alanmatys/

# In[1]:


# Import necesary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


# In[2]:


# We define the module path in order to be able to call the .py functions in the module
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)


# In[3]:


# We iimport the functions to read the csv file (read_df), add the synthetic_features (synthetic_features), and the geo data(geo_data)
# To see the functions look for the make_data/input_data.py
from make_data.input_data import read_df, synthetic_features, geo_data


# In[4]:


# We read the csv file
df = read_df()
df.head()


# In this case it's just one row on the flight number, so we will skip it for the time being

# # 2) Synthetic Features
#
# We add the requested features with their specific criteria:
# 1) high_season : 1 if Date-I is between Dec-15 and Mar-3, or Jul-15 and Jul-31, or Sep-11 and Sep-30, 0 otherwise.
#
# 2) min_diff : difference in minutes between Date-O and Date-I .
#
# 3) delay_15 : 1 if min_diff > 15, 0 if not.
#
# 4) period_day : morning (between 5:00 and 11:59), afternoon (between 12:00 and 18:59) and night (between 19:00 and 4:59), based
# onDate-I .

# In[5]:


df = synthetic_features(df)
df.head()


# ## Extra Features
#
# We will add some features from the dataset and exogenous sources.
#
# - distance: geodesic distance between the origin and the destination, using geopy data
# - country_des: country of destination

# In[6]:


df = geo_data(df)
df.head()


# ## 1) EDA (Exploratory Data Analysis)
#
# In this section we try to describe the categorical and the numerical variables

# ### Categorical Features

# In[7]:


def programed_vs_operated(df: pd.DataFrame):
    """

    Plot the comparision of the Programed features vs the Operated features

    Parameters
    --------------------

    - df: pandas.DataFrame
        Pandas DataFrame containing the this columns:
        - Origin Variables (Ori-I/Ori-O)
        - Destination Variables (Des-I/Des-O)

    """
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 12))

    fig.suptitle("Programmed vs Operation")

    sns.countplot(
        x=df["Ori-I"],
        order=df["Ori-I"].value_counts().index,
        ax=axes[0, 0],
        color="red",
        alpha=0.7,
    )

    sns.countplot(
        x=df["Ori-O"],
        order=df["Ori-O"].value_counts().index,
        ax=axes[0, 1],
        color="blue",
        alpha=0.7,
    )

    sns.countplot(
        x=df["Des-I"],
        order=df["Des-I"].value_counts().index,
        ax=axes[1, 0],
        color="red",
        alpha=0.7,
    )
    axes[1, 0].tick_params(axis="x", rotation=90)

    sns.countplot(
        x=df["Des-O"],
        order=df["Des-O"].value_counts().index,
        ax=axes[1, 1],
        color="blue",
        alpha=0.7,
    )
    axes[1, 1].tick_params(axis="x", rotation=90)

    dis_des = round((len(df[df["Des-I"] != df["Des-O"]]) / len(df)) * 100, 2)

    axes[1, 1].text(
        0.4,
        0.70,
        f"{dis_des}% of the flights that ended on a different destination",
        ha="left",
        va="top",
        color="blue",
        transform=axes[1, 1].transAxes,
    )

    sns.countplot(
        x=df["Emp-I"],
        order=df["Emp-I"].value_counts().index,
        ax=axes[2, 0],
        color="red",
        alpha=0.7,
    )
    axes[2, 0].tick_params(axis="x", rotation=90)

    sns.countplot(
        x=df["Emp-O"],
        order=df["Emp-O"].value_counts().index,
        ax=axes[2, 1],
        color="blue",
        alpha=0.7,
    )
    axes[2, 1].tick_params(axis="x", rotation=90)

    dis_emp = round((len(df[df["Emp-I"] != df["Emp-O"]]) / len(df)) * 100, 2)

    axes[2, 1].text(
        0.4,
        0.70,
        f"{dis_emp}% of the flights that have a diifferent Airline Code",
        ha="left",
        va="top",
        color="blue",
        transform=axes[2, 1].transAxes,
    )

    plt.show()


# In[8]:


programed_vs_operated(df)


# From this chart we can take the following insights
# - More than 1/4 flights had a different airline code from the one that was programed.
# - There are some flights, not usually, that ended on a different destination, based on the destination code

# In[10]:


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
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 6))

    sns.histplot(
        x=df["TIPOVUELO"],
        multiple="dodge",
        stat="percent",
        shrink=0.8,
        common_norm=False,
        ax=axes[0, 0],
    )

    sns.histplot(
        x=df["DIANOM"],
        multiple="dodge",
        stat="percent",
        shrink=0.8,
        ax=axes[0, 1],
        discrete=True,
        hue=df["TIPOVUELO"],
    )

    axes[0, 1].get_legend().remove()

    sns.countplot(
        x=df["SIGLADES"],
        ax=axes[1, 0],
        order=df["SIGLADES"].value_counts().index,
        hue=df["TIPOVUELO"],
    )

    axes[1, 0].tick_params(axis="x", rotation=90)
    axes[1, 0].set(xlabel=None)

    sns.histplot(
        x=df["OPERA"],
        multiple="dodge",
        stat="percent",
        shrink=0.8,
        ax=axes[1, 1],
        discrete=True,
        hue=df["TIPOVUELO"],
    )

    axes[1, 1].tick_params(axis="x", rotation=90)
    axes[1, 1].set(xlabel=None)

    plt.show()


# In[11]:


other_categorical(df)


# From this chart we can take the following insights:
# - FROM Chart 1:
#
# There are more National Flights (within Chile) than International Flights (from SCL to another country), but more or less is equivalent.
#
# - From Chart 2:
#
# International flights are uniform in terms of the day of the flight, but for National flights it's not the same, there are significantly fewer flights on Saturdays and the proportion changes on this day (More Int flights)
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

# In[12]:


def time_s_airlines(df, var):
    """ """
    fig, ax = plt.subplots(nrows=4, figsize=(20, 28))

    plot = df.groupby([df["Fecha-I"].dt.date, df[var]]).agg(vuelos=("Vlo-I", "count"))

    sns.lineplot(plot, x="Fecha-I", y="vuelos", hue=var, ax=ax[0])

    ax[0].set_title("Flights per Day")
    ax[0].set(xlabel=None)

    sns.move_legend(
        ax[0], loc="center left", bbox_to_anchor=(1, 0.7), ncol=2, title=None
    )

    plot_2 = (
        df[~df.OPERA.isin(["Grupo LATAM", "Sky Airline"])]
        .groupby([df["Fecha-I"].dt.date, df[var]])
        .agg(vuelos=("Vlo-I", "count"))
    )

    sns.lineplot(plot_2, x="Fecha-I", y="vuelos", hue=var, ax=ax[1])

    sns.move_legend(
        ax[1], loc="center left", bbox_to_anchor=(1, 0.7), ncol=2, title=None
    )

    ax[1].set_title("Flights Excluding LATAM & Sky")
    ax[1].set(xlabel=None)

    df_nac = df[df["TIPOVUELO"] == "N"]

    plot_nac = df_nac.groupby([df_nac["Fecha-I"].dt.date, df[var]]).agg(
        vuelos=("Vlo-I", "count")
    )

    sns.lineplot(plot_nac, x="Fecha-I", y="vuelos", hue=var, ax=ax[2])

    sns.move_legend(
        ax[2],
        loc="center left",
        bbox_to_anchor=(1, 0.9),
        ncol=2,
        title=None,
        fancybox=True,
        shadow=True,
    )

    ax[2].set_title("Nac Flights")
    ax[2].set(xlabel=None)

    df_int = df[df["TIPOVUELO"] == "I"]

    plot_int = df_int.groupby([df_int["Fecha-I"].dt.date, df[var]]).agg(
        vuelos=("Vlo-I", "count")
    )

    sns.lineplot(plot_int, x="Fecha-I", y="vuelos", hue=var)

    ax[3].set_title("Int Flights")
    ax[3].set(xlabel=None)

    sns.move_legend(
        ax[3], loc="center left", bbox_to_anchor=(1, 0.7), ncol=2, title=None
    )

    plt.show()


# In[13]:


time_s_airlines(df, "OPERA")


# In[14]:


time_s_airlines(df, "SIGLADES")


# In[15]:


time_s_airlines(df, "country_des")


# ## Numerical Features
#
# We try some plots on the Numerical Features

# In[46]:


def plot_numeric(df):
    """ """

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 4))

    sns.displot(x=df["distance"], hue=df["OPERA"], bins=12, ax=axes[0])

    plt.show()


# In[47]:


plot_numeric(df)


# # 3) Examine the Behaviour

# In[21]:


plt.figure(figsize=(20, 10))
sns.heatmap(df.corr(method="spearman"), annot=True, cmap="coolwarm")


# In[12]:


def add_insights(df):
    """ """

    df["delay"] = np.where(df["min_diff"] > 0, 1, 0)

    df["diff_code"] = np.where(df["Des-I"] != df["Des-O"], 1, 0)

    df["top_airline"] = np.where(df["OPERA"].isin(["Grupo LATAM"]), 1, 0)

    df["top_destination_int"] = np.where(df["country_des"].isin(["Perú"]), 1, 0)

    return df


# In[ ]:


df = add_insights(df)


# # 4) Train the models
