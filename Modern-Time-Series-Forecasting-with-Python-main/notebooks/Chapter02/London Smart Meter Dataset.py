#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Changing the working directory to the root
get_ipython().run_line_magic('cd', '../..')


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
import plotly.io as pio
pio.templates.default = "plotly_white"
import pandas as pd
from pathlib import Path
from tqdm.autonotebook import tqdm
# %load_ext autoreload
# %autoreload 2
np.random.seed()
tqdm.pandas()


# # Reading data to pandas dataframe
# 
# Let's use London Smart Meters dataset from Kaggle for the current exercise. The source of the data is from London Data Store and Jean-Michel D. has augumented the dataset with a few more details and packaged it as a Kaggle Dataset.
# 
# ## About the Dataset
# 
# London Data Store, free and open data-sharing portal, provided this dataset, which was collected and enriched by Jean-Michel D and uploaded on Kaggle @ https://www.kaggle.com/jeanmidev/smart-meters-in-london. 
# The dataset has energy consumption readings for a sample of 5,567 London Households that took part in the UK Power Networks led Low Carbon London project between November 2011 and February 2014. Readings were taken at half hourly intervals. Some metadata about the households are also available as part of the dataset.
# 
# ## Data Wrangling
# The Kaggle dataset also has the time series data preprocessed on a daily level and combined all the separate files, etc. But let’s ignore those files and start with the raw files, which is in the hhblock_dataset folder.

# In[7]:


os.makedirs("imgs/chapter_2", exist_ok=True)
source_data = Path("data/london_smart_meters/")
block_data_path = source_data/"hhblock_dataset"/"hhblock_dataset"


# In[8]:


assert block_data_path.is_dir(), "Please check if the dataset has been downloaded properly. Refer to the Preface of the book or the Readme in the repo for expected data"


# ### Converting the half hourly block level dataset into a time series data 
# 
# Let's pick one block and see how we can transform the data.

# In[4]:


block_1 = pd.read_csv(block_data_path/"block_0.csv", parse_dates=False)

block_1['day'] = pd.to_datetime(block_1['day'], yearfirst=True)

block_1.head()


# In[5]:


#Check End Dates of all time series
block_1.groupby("LCLid")['day'].max().sample(5)


# In[6]:


max_date = None
for f in tqdm(block_data_path.glob("*.csv")):
    df = pd.read_csv(f, parse_dates=False)
    df['day'] = pd.to_datetime(df['day'], yearfirst=True)
    if max_date is None:
        max_date = df['day'].max()
    else:
        if df['day'].max()>max_date:
            max_date = df['day'].max()
print(f"Max Date across all blocks: {max_date}")
del df


# In[7]:


#Reshaping the dataframe into the long form with hour blocks along the rows
block_1 = block_1.set_index(['LCLid', "day"]).stack().reset_index().rename(columns={"level_2": "hour_block", 0: "energy_consumption"})
#Creating a numerical hourblock column
block_1['offset'] = block_1['hour_block'].str.replace("hh_", "").astype(int)

block_1.head()


# #### Compact Form

# In[8]:


def preprocess_compact(x):
    start_date = x['day'].min()
    name = x['LCLid'].unique()[0]
    ### Fill missing dates with NaN ###
    # Create a date range from  min to max
    dr = pd.date_range(start=x['day'].min(), end=max_date, freq="1D")
    # Add hh_0 to hh_47 to columns and with some unstack magic recreating date-hh_x combinations
    dr = pd.DataFrame(columns=[f"hh_{i}" for i in range(48)], index=dr).unstack().reset_index()
    # renaming the columns
    dr.columns = ["hour_block", "day", "_"]
    # left merging the dataframe to the standard dataframe
    # now the missing values will be left as NaN
    dr = dr.merge(x, on=['hour_block','day'], how='left')
    # sorting the rows
    dr.sort_values(['day',"offset"], inplace=True)
    # extracting the timeseries array
    ts = dr['energy_consumption'].values
    len_ts = len(ts)
    return start_date, name, ts, len_ts


# In[9]:


def load_process_block_compact(block_df, freq="30min", ts_identifier="series_name", value_name="series_value"):
    grps = block_df.groupby('LCLid')
    all_series = []
    all_start_dates = []
    all_names = []
    all_data = {}
    all_len = []
    for idx, df in tqdm(grps, leave=False):
        start_date, name, ts, len_ts = preprocess_compact(df)
        all_series.append(ts)
        all_start_dates.append(start_date)
        all_names.append(name)
        all_len.append(len_ts)

    all_data[ts_identifier] = all_names
    all_data['start_timestamp'] = all_start_dates
    all_data['frequency'] = freq
    all_data[value_name] = all_series
    all_data['series_length'] = all_len
    return pd.DataFrame(all_data)

block1_compact = load_process_block_compact(block_1, freq="30min", ts_identifier="LCLid", value_name="energy_consumption")


# In[10]:


block1_compact.head()


# In[11]:


display(block1_compact.memory_usage(deep=True))
print(f"Total: {block1_compact.memory_usage(deep=True).sum()/1024**2} MB")


# #### Expanded Form

# In[12]:


def preprocess_expanded(x):
    start_date = x['day'].min()
    ### Fill missing dates with NaN ###
    # Create a date range from  min to max
    dr = pd.date_range(start=x['day'].min(), end=x['day'].max(), freq="1D")
    # Add hh_0 to hh_47 to columns and with some unstack magic recreating date-hh_x combinations
    dr = pd.DataFrame(columns=[f"hh_{i}" for i in range(48)], index=dr).unstack().reset_index()
    # renaming the columns
    dr.columns = ["hour_block", "day", "_"]
    # left merging the dataframe to the standard dataframe
    # now the missing values will be left as NaN
    dr = dr.merge(x, on=['hour_block','day'], how='left')
    dr['series_length'] = len(dr)
    return dr


# In[13]:


def load_process_block_expanded(block_df, freq="30min"):
    grps = block_df.groupby('LCLid')
    all_series = []
    for idx, df in tqdm(grps, leave=False):
        ts = preprocess_expanded(df)
        all_series.append(ts)

    block_df = pd.concat(all_series)
    # Recreate Offset because there would be null rows now
    block_df['offset'] = block_df['hour_block'].str.replace("hh_", "").astype(int)
    # Creating a datetime column with the date | Will take some time because operation is not vectorized
    block_df['timestamp'] = block_df['day'] + block_df['offset']*30*pd.offsets.Minute()
    block_df['frequency'] = freq
    block_df.sort_values(["LCLid","timestamp"], inplace=True)
    block_df.drop(columns=["_", "hour_block", "offset", "day"], inplace=True)
    return block_df
#     del all_series
block1_expanded = load_process_block_expanded(block_1, freq="30min")


# In[14]:


block1_expanded.head()


# In[15]:


display(block1_expanded.memory_usage())
print(f"Total: {block1_expanded.memory_usage().sum()/1024**2} MB")


# In[16]:


del block1_expanded, block_1, block1_compact


# #### Reading and combining all the block data into a single dataframe

# In[17]:


block_df_l = []
for file in tqdm(sorted(list(block_data_path.glob("*.csv"))), desc="Processing Blocks.."):
    block_df = pd.read_csv(file, parse_dates=False)
    block_df['day'] = pd.to_datetime(block_df['day'], yearfirst=True)
    # Taking only from 2012-01-01
    block_df = block_df.loc[block_df['day']>="2012-01-01"]
    #Reshaping the dataframe into the long form with hour blocks along the rows
    block_df = block_df.set_index(['LCLid', "day"]).stack().reset_index().rename(columns={"level_2": "hour_block", 0: "energy_consumption"})
    #Creating a numerical hourblock column
    block_df['offset'] = block_df['hour_block'].str.replace("hh_", "").astype(int)
    block_df_l.append(load_process_block_compact(block_df, freq="30min", ts_identifier="LCLid", value_name="energy_consumption"))


# In[18]:


hhblock_df = pd.concat(block_df_l)
del block_df_l
display(hhblock_df.memory_usage(deep=True))
print(f"Total: {hhblock_df.memory_usage(deep=True).sum()/1024**2} MB")


# In[19]:


hhblock_df.head()


# ### Merging additional information

# #### Household information

# In[20]:


household_info = pd.read_csv(source_data/"informations_households.csv")
household_info.head()


# In[21]:


hhblock_df = hhblock_df.merge(household_info, on='LCLid', validate="one_to_one")
hhblock_df.head()


# #### Weather and Bank Holidays

# In[22]:


bank_holidays = pd.read_csv(source_data/"uk_bank_holidays.csv", parse_dates=False)
bank_holidays['Bank holidays'] = pd.to_datetime(bank_holidays['Bank holidays'], yearfirst=True)
bank_holidays.set_index("Bank holidays", inplace=True)
bank_holidays.head()


# In[23]:


#Reindex on standard date range
bank_holidays = bank_holidays.resample("30min").asfreq()
bank_holidays = bank_holidays.groupby(bank_holidays.index.date).ffill().fillna("NO_HOLIDAY")
bank_holidays.index.name="datetime"
bank_holidays.head()


# In[24]:


weather_hourly = pd.read_csv(source_data/"weather_hourly_darksky.csv", parse_dates=False)
weather_hourly['time'] = pd.to_datetime(weather_hourly['time'], yearfirst=True)
weather_hourly.set_index("time", inplace=True)
weather_hourly.head()


# In[25]:


#Resampling at 30min and forward fill
weather_hourly = weather_hourly.resample("30min").ffill()
weather_hourly.head()


# In[26]:


def map_weather_holidays(row):
    date_range = pd.date_range(row['start_timestamp'], periods=row['series_length'], freq=row['frequency'])
    std_df = pd.DataFrame(index=date_range)
    #Filling Na iwth NO_HOLIDAY cause rows before earliers holiday will be NaN
    holidays = std_df.join(bank_holidays, how="left").fillna("NO_HOLIDAY")
    weather = std_df.join(weather_hourly, how='left')
    assert len(holidays)==row['series_length'], "Length of holidays should be same as series length"
    assert len(weather)==row['series_length'], "Length of weather should be same as series length"
    row['holidays'] = holidays['Type'].values
    for col in weather:
        row[col] = weather[col].values
    return row


# In[27]:


hhblock_df = hhblock_df.progress_apply(map_weather_holidays, axis=1)

hhblock_df.head()


# In[28]:


del block_df, weather_hourly, bank_holidays, household_info


# In[29]:


display(hhblock_df.memory_usage(deep=True))
print(f"Total: {hhblock_df.memory_usage(deep=True).sum()/1024**2} MB")


# ### Saving the file on disk

# Saving the entire dataset in a ts file

# In[ ]:


os.makedirs("data/london_smart_meters/preprocessed", exist_ok=True)


# In[34]:


#Takes a long time to finish. Comment out and execute only if needed
# from src.utils.data_utils import write_compact_to_ts
# write_compact_to_ts(hhblock_df,
#        static_columns = ['LCLid', 'start_timestamp', 'frequency','series_length', 'stdorToU', 'Acorn', 'Acorn_grouped', 'file'], 
#        time_varying_columns = ['energy_consumption', 'holidays', 'visibility', 'windBearing', 'temperature', 'dewPoint',
#                               'pressure', 'apparentTemperature', 'windSpeed', 'precipType', 'icon','humidity', 'summary'],
#        filename=f"data/london_smart_meters/preprocessed/london_smart_meters_merged.ts",
#        sep=";",
#       chunk_size=1000)


# In[33]:


# Saving the LCLid - Acorn map as a pickle to be used later
hhblock_df[['LCLid',"file", "Acorn_grouped"]].to_pickle(f"data/london_smart_meters/preprocessed/london_smart_meters_lclid_acorn_map.pkl")


# Saving blocks in 8 chunks as parquet

# In[35]:


# Splitting the blocks into 8 chunks
blocks = [f"block_{i}" for i in range(111)]

n_chunks= 8
split_blocks = [blocks[i:i + n_chunks] for i in range(0, len(blocks), n_chunks)] 


# In[36]:


#Writing each chunk to disk
for blk in tqdm(split_blocks):
    df = hhblock_df.loc[hhblock_df.file.isin(blk)]
    blk = [int(b.replace("block_","")) for b in blk]
    block_str = f"block_{min(blk)}-{max(blk)}"
    df.to_parquet(f"data/london_smart_meters/preprocessed/london_smart_meters_merged_{block_str}.parquet")

