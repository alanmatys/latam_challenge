#!/usr/bin/env python
# coding: utf-8

# In[170]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os


# In[171]:


def read_df():
    """ """

    df = pd.read_csv(
        os.path.join("..", "data", "raw", "dataset_SCL.csv"),
        parse_dates=["Fecha-I", "Fecha-O"],
    )

    # NaN check in data frame
    if df.isnull().any().any():
        print("There is/are NaN in these Rows\n", df.isnull().any())

    return df


# In[172]:


df = read_df()
df.head()


# In[204]:


condition = (
    (df["Fecha-I"].between("2016-12-15", "2017-03-03"))
    | (df["Fecha-I"].between("2017-09-11", "2017-09-30"))
    | (df["Fecha-I"].between("2017-07-15", "2017-07-31"))
)


# In[208]:


def synthetic_features(df):
    """ """

    new_cols = ["high_season", "min_diff", "delay_15"]

    condition = (
        (df["Fecha-I"].between("2016-12-15", "2017-03-03"))
        | (df["Fecha-I"].between("2017-09-11", "2017-09-30"))
        | (df["Fecha-I"].between("2017-07-15", "2017-07-31"))
    )

    df[new_cols[0]] = np.where(condition, 1, 0)

    df[new_cols[1]] = (df["Fecha-O"] - df["Fecha-I"]).dt.total_seconds() / 60.0

    df[new_cols[2]] = np.where(df["min_diff"] > 15, 1, 0)

    return df


# In[206]:


df = synthetic_features(df)
df.head()


# ## EDA (Exploratory Data Analysis)
#
# In this section we try to describe the categorical and the numerical variables

# ### Categorical Variables

# In[108]:


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

dis_des = len(df[df["Des-I"] != df["Des-O"]])

axes[1, 1].text(
    0.4,
    0.70,
    f"There are {dis_des} flights that ended on a different destination",
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

dis_emp = len(df[df["Emp-I"] != df["Emp-O"]])

axes[2, 1].text(
    0.4,
    0.70,
    f"There are {dis_emp} flights that have a diifferent Airline Code",
    ha="left",
    va="top",
    color="blue",
    transform=axes[2, 1].transAxes,
)

plt.show()


# In[106]:


df[df["Emp-I"] != df["Emp-O"]]
