import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import plotly.express as px
import calmap
import streamlit as st
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import emoji

sns.set_theme(style='dark')

base_dir = os.path.dirname(__file__)
zip_path = os.path.join(base_dir, 'all_data.zip')

all_df = pd.read_csv(zip_path, compression='zip')

# Menyortir dan memastikan kolom datetime
datetime_columns = ['datetime']
all_df.sort_values(by='datetime', inplace=True)
all_df.reset_index(drop=True, inplace=True)

for column in datetime_columns:
  all_df[column] = pd.to_datetime(all_df[column])

min_date = all_df['datetime'].min().date()
max_date = all_df['datetime'].max().date()

st.set_page_config(layout='wide')

# Filter Tanggal, Statiun, Polutan, dan Periode di Sidebar
with st.sidebar:
  st.sidebar.header(':calendar: Date')
  start_date = st.date_input('Start Date', value=min_date)
  end_date = st.date_input('End Date', value=max_date)

  st.sidebar.header(':cityscape: Station')
  stations = all_df['station'].unique().tolist()
  choose_stations = st.sidebar.multiselect(
    label = 'Choose Stations',
    options = ['All Stations'] + stations,
    default = ['All Stations'],
    label_visibility = 'collapsed'
  )
  if 'All Stations' in choose_stations:
    selected_stations = stations  # semua polutan dipilih
  else:
    selected_stations = choose_stations  # hanya yang dipilih

  st.sidebar.header(':fog: Pollutant')
  pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
  choose_pollutants = st.sidebar.multiselect(
    label = 'Choose Pollutants',
    options = ['All Pollutants'] + pollutants,
    default = ['All Pollutants'],
    label_visibility = 'collapsed'
  )
  if 'All Pollutants' in choose_pollutants:
    selected_pollutants = pollutants  # semua polutan dipilih
  else:
    selected_pollutants = choose_pollutants  # hanya yang dipilih

  st.sidebar.header(':stopwatch: Period')
  selected_period = st.selectbox(
    label = 'Choose Period',
    options = ['Hourly', 'Daily', 'Weekly', 'Monthly'],
    index = 1,
    label_visibility = 'collapsed'
  )
  resample_map = {
    'Hourly': 'H',
    'Daily': 'D',
    'Weekly': 'W',
    'Monthly': 'M'
  }
  resample_freq = resample_map[selected_period]

if start_date > end_date:
  st.error('Oops! Start Date can't be later than the End Date.')

main_df = all_df[(all_df['datetime'].dt.date >= start_date) &
                 (all_df['datetime'].dt.date <= end_date)]

st.header('Air Quality Dashboard :mag:')

st.subheader('Weather Condition')
weather_df = main_df[main_df['station'].isin(selected_stations)].copy()

avg_temp = weather_df['TEMP'].mean()
avg_pres = weather_df['PRES'].mean()
avg_dewp = weather_df['DEWP'].mean()
avg_rain = weather_df['RAIN'].mean()
avg_ws = weather_df['WSPM'].mean()

def deg_to_compass(degree):
  if pd.isna(degree):
    return 'N/A'
  directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
  idx = int((degree/22.5) + 0.5) % 16
  return directions[idx]

def centered_metric(label, value, unit=''):
    st.markdown(f"""
        <div style='
          display: flex; 
          flex-direction: column; 
          align-items: center; 
          font-family: "Segoe UI Emoji", "Noto Color Emoji", sans-serif;
          text-align: center;
          padding: 10px;
        '>
          <div style='font-size: 20px; font-weight: bold; margin-bottom: 4px;'>{label}</div>
          <div style='font-size: 28px; font-weight: 500; text-align: center;'>{value} {unit}</div>
        </div>
    """, unsafe_allow_html=True)

if 'wd_deg' in weather_df.columns and not weather_df['wd_deg'].dropna().empty:
  mode_wd_deg = weather_df['wd_deg'].mode().iloc[0]
  compass_wd = deg_to_compass(mode_wd_deg)
else:
  compass_wd = 'N/A'
  
col11, col12, col13 = st.columns(3)
with col11:
  centered_metric(emoji.emojize('TEMP :thermometer:'), f'{avg_temp:.2f}', '°C')
with col12:
  centered_metric(emoji.emojize('PRESS :cyclone:'), f'{avg_pres:.2f}', 'hPa')
with col13:
  centered_metric(emoji.emojize('DEWP :droplet:'), f'{avg_dewp:.2f}', '°C')

col21, col22, col23 = st.columns(3)
with col21:
  centered_metric(emoji.emojize('RAIN :cloud_with_rain:'), f'{avg_rain:.2f}', 'mm')
with col22:
  centered_metric(emoji.emojize('WIND DIRECTION :compass:'), compass_wd)
with col23:
  centered_metric(emoji.emojize('WIND SPEED :dash:', language='alias'), f'{avg_ws:.2f}', 'm/s')

tab1, tab2, tab3 = st.tabs([
    ':bar_chart: Air Quality Index (AQI)', 
    ':chart_with_upwards_trend: Pollutant Trends', 
    ':clock3: Diurnal Patterns'
])

with tab1:
  # st.subheader(':bar_chart: Air Quality Index (AQI)')
  col1, col2 = st.columns(2)

  filtered_df = main_df[main_df['station'].isin(selected_stations)].copy()
  filtered_df['datetime'] = pd.to_datetime(filtered_df[['year', 'month', 'day', 'hour']])
  filtered_df.set_index('datetime', inplace=True)

  aqi_resampled = filtered_df['AQI_CN'].resample(resample_freq).max()
  x_label = 'Date and Time'
  if selected_period == 'Hourly':
    title_label = f'{selected_period} Air Quality Index (AQI) Trend ({start_date.strftime('%d %b %Y')})'
  else:
    tititle_label = f'{selected_period} Air Quality Index (AQI) Trend ({start_date.strftime('%d %b %Y')} - {end_date.strftime('%d %b %Y')})'
  
  with col1:
    st.subheader('AQI Trend') #st.markdown('#### AQI Trend')
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(aqi_resampled.index, aqi_resampled, color='red', linewidth=1.5)
    ax.set_title(title_label, fontsize=18)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel('AQI Value', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True)

    plt.tight_layout()
    st.pyplot(fig)
    
  with col2:
    st.subheader('AQI Category Distribution') #st.markdown('#### AQI Category Distribution')
    aqi_resampled = filtered_df['AQI_CN'].resample(resample_freq).max()
    aqi_categories = [
      (0, 50, 'Excellent'),
      (51, 100, 'Good'),
      (101, 150, 'Slight Pollution'),
      (151, 200, 'Moderate Pollution'),
      (201, 300, 'Heavy Pollution'),
      (301, 500, 'Severe Pollution')
    ]
    kategori_counts = []
    kategori_labels = []
    for low, high, label in aqi_categories:
        count = aqi_resampled[(aqi_resampled >= low) & (aqi_resampled <= high)].shape[0]
        kategori_counts.append(count)
        kategori_labels.append(f'{low}–{high}')

    colors = ['green', 'yellow', 'orange', 'red', 'purple', 'maroon']

    custom_legend = [
      Patch(facecolor=color, label=label)
      for (_, _, label), color in zip(aqi_categories, colors)
    ]

    fig, ax = plt.subplots(figsize=(10,6))
    bars = ax.bar(kategori_labels, kategori_counts, color=colors)
    ax.set_title(f'AQI Category Distribution ({selected_period})', fontsize=18)
    ax.set_xlabel('AQI Value', fontsize=16)
    ax.set_ylabel('Data Count', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(handles=custom_legend, title='AQI Categories', fontsize=14, title_fontsize=14)

    for idx, bar in enumerate(bars):
      height = bar.get_height()
      ax.text(bar.get_x() + bar.get_width()/2, height + 0.1, str(height),
              ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

with tab2:
  # st.subheader(':chart_with_upwards_trend: Pollutant Trends')
  col1, col2 = st.columns(2)
  with col1:
    #st.subheader('Pollutant Concentration Demographics') #st.markdown('#### Pollutant Concentration Demographics')
    st.subheader('Demographics by Station') #st.markdown('#### Demographics by Station')
    if selected_stations and selected_pollutants:
      filtered_df = main_df[main_df['station'].isin(selected_stations)].copy()
      pollutants_agg_df = filtered_df.groupby(by='station')[selected_pollutants].mean()
      pollutants_agg_df['total_average'] = pollutants_agg_df.sum(axis=1)

      max_station = pollutants_agg_df['total_average'].idxmax()
      min_station = pollutants_agg_df['total_average'].idxmin()
      
      colors = []
      for station in pollutants_agg_df.index:
        if station == max_station:
          colors.append('red')
        elif station == min_station:
          colors.append('green')
        else:
          colors.append('lightgray')

      fig, ax = plt.subplots(figsize=(10,6))
      pollutants_agg_df['total_average'].plot(kind='bar', color=colors, ax=ax)

      ax.set_title('Total Average Pollutant Concentration per Station', fontsize=18)
      ax.set_xlabel('Station', fontsize=16)
      ax.set_ylabel('Total Average Concentration (µg/m³)', fontsize=16)
      ax.tick_params(axis='both', labelsize=14)
      ax.grid(axis='y', linestyle='--', alpha=0.7)
      plt.xticks(rotation=45)
      
      plt.tight_layout()
      st.pyplot(fig)

      df_to_display = pollutants_agg_df.drop(columns=['total_average'])
      with st.expander('Lihat Data Rata-Rata per Stasiun'):
        st.dataframe(df_to_display)
    
    elif not selected_stations:
      st.warning('Please select at least one station to continue.')

    elif not selected_pollutants:
      st.warning('Please select at least one pollutant type.')

  with col2:
    st.subheader('Pollutant Concentration Trend') #st.markdown('#### Pollutant Concentration Trend')
    if selected_stations and selected_pollutants:      
      filtered_df = main_df[main_df['station'].isin(selected_stations)].copy()
      filtered_df['datetime'] = pd.to_datetime(filtered_df[['year', 'month', 'day', 'hour']])
      filtered_df.set_index('datetime', inplace=True)

      pollutant_resampled_df = filtered_df[selected_pollutants].resample(resample_freq).mean()
      x_label = 'Date and Time'
      if selected_period == 'Hourly':
        title_label = f'{selected_period} Pollutant Concentration Trend ({start_date.strftime('%d %b %Y')})'
      else:
        title_label = f'{selected_period} Pollutant Concentration Trend ({start_date.strftime('%d %b %Y')} - {end_date.strftime('%d %b %Y')})'

      pollutant_resampled_df['total_average'] = pollutant_resampled_df.sum(axis=1)

      fig, ax = plt.subplots(figsize=(10,6))

      all_pollutants_set = set(['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'])
      selected_pollutants_set = set(selected_pollutants)

      if selected_pollutants_set == all_pollutants_set:
        ax.plot(
          pollutant_resampled_df.index, 
          pollutant_resampled_df['total_average'], 
          label='All Pollutants Trend', 
          color='black', 
          linewidth=2.5)
      else:
        label = 'Selected Pollutants Trend' if len(selected_pollutants) > 1 else f'{selected_pollutants[0]} Trend'
        ax.plot(
          pollutant_resampled_df.index, 
          pollutant_resampled_df['total_average'], 
          label=label, 
          color='blue', 
          linewidth=2.5)

      ax.set_title(title_label, fontsize=18)
      ax.set_xlabel(x_label, fontsize=16)
      ax.set_ylabel('Total Average Concentration (µg/m³)', fontsize=16)
      ax.tick_params(axis='both', labelsize=14)
      ax.legend(fontsize=14)
      ax.grid(True, linestyle='--', alpha=0.5)
      
      plt.tight_layout()
      st.pyplot(fig)

    elif not selected_stations:
      st.warning('Please select at least one station to continue.')

    elif not selected_pollutants:
      st.warning('Please select at least one pollutant type.')

with tab3:
  # st.subheader(':clock3: Diurnal Patterns')
  col1, col2 = st.columns(2)
  with col1:
    st.subheader('All Day') #st.markdown('#### Diurnal Pattern All Day')
    if selected_stations and selected_pollutants:
      filtered_df = main_df[main_df['station'].isin(selected_stations)].copy()
      filtered_df['datetime'] = pd.to_datetime(filtered_df[['year', 'month', 'day', 'hour']])
      filtered_df['hour'] = filtered_df['datetime'].dt.hour

      all_pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
      selected_pollutants_set = set(selected_pollutants)
      all_pollutants_set = set(all_pollutants)

      fig, ax = plt.subplots(figsize=(10,6))

      # Kondisi Awal = Pilih Satu Hari
      if start_date == end_date:
        if selected_pollutants_set == all_pollutants_set:
          filtered_df['total_pollutant'] = filtered_df[all_pollutants].sum(axis=1)
          sns.lineplot(data=filtered_df, x='hour', y='total_pollutant', marker='o', ax=ax)
          ax.set_title(f'Diurnal Pattern of Total Pollutant Concentration ({start_date})', fontsize=18)
        elif len(selected_pollutants) > 1:
          filtered_df['total_pollutant'] = filtered_df[selected_pollutants].sum(axis=1)
          sns.lineplot(data=filtered_df, x='hour', y='total_pollutant', marker='o', ax=ax)
          ax.set_title(f'Diurnal Pattern of Selected Pollutants Concentration ({start_date})', fontsize=18)
        else:
          polutan = selected_pollutants[0]
          sns.lineplot(data=filtered_df, x='hour', y=polutan, marker='o', ax=ax)
          ax.set_title(f'Diurnal Pattern of {polutan} Concentration ({start_date})', fontsize=18)
      else:
        # Kondisi 1: Pilih Semua Polutan
        if selected_pollutants_set == all_pollutants_set:
          filtered_df['total_pollutant'] = filtered_df[all_pollutants].sum(axis=1)
          diurnal_avg = filtered_df.groupby('hour')['total_pollutant'].mean().reset_index()
          sns.lineplot(data=diurnal_avg, x='hour', y='total_pollutant', marker='o', ax=ax)
          ax.set_title('Diurnal Pattern of All Pollutant Concentration', fontsize=18)

        # Kondisi 2: Pilih Beberapa Polutan
        elif len(selected_pollutants) > 1:
          filtered_df['total_pollutant'] = filtered_df[selected_pollutants].sum(axis=1)
          diurnal_avg = filtered_df.groupby('hour')['total_pollutant'].mean().reset_index()
          sns.lineplot(data=diurnal_avg, x='hour', y='total_pollutant', marker='o', ax=ax)
          ax.set_title('Diurnal Pattern of Selected Pollutant Concentration', fontsize=18)

        # Kondisi 3: Pilih Satu Polutan
        else:
          pollutant = selected_pollutants[0]
          diurnal_avg = filtered_df.groupby('hour')[pollutant].mean().reset_index()
          sns.lineplot(data=diurnal_avg, x='hour', y=pollutant, marker='o', ax=ax)
          ax.set_title(f'Diurnal Pattern of {pollutant} Concentration', fontsize=18)

      # Format sumbu & tampilan
      ax.set_xlabel('Hour of Day', fontsize=16)
      ax.set_ylabel('Total Average Concentration (µg/m³)', fontsize=16)
      ax.set_xticks(range(0, 24))
      ax.tick_params(axis='both', labelsize=14)
      ax.grid(True)
      
      plt.tight_layout()
      st.pyplot(fig)

      # Heatmap: Hour vs Day of the Month
      with st.expander('View Heatmap: Hour vs Day of the Month'):
        if 'total_pollutant' not in filtered_df.columns:
          if selected_pollutants_set == all_pollutants_set:
            filtered_df['total_pollutant'] = filtered_df[all_pollutants].sum(axis=1)
          elif len(selected_pollutants) > 1:
            filtered_df['total_pollutant'] = filtered_df[selected_pollutants].sum(axis=1)
          else:
            pollutant = selected_pollutants[0]
            filtered_df['total_pollutant'] = filtered_df[pollutant]

        filtered_df['day'] = filtered_df['datetime'].dt.day

        heatmap_data = filtered_df.groupby(['day', 'hour'])['total_pollutant'].mean().unstack()

        fig_hm, ax_hm = plt.subplots(figsize=(12, 6))
        sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax_hm, linewidths=0.5, linecolor='white')
        ax_hm.set_title('Heatmap: Total Average Concentration per Hour and Day of the Month', fontsize=18)
        ax_hm.set_xlabel('Hour of Day', fontsize=16)
        ax_hm.set_ylabel('Day of the Month', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)

        plt.tight_layout()
        st.pyplot(fig_hm)
        
    elif not selected_stations:
      st.warning('Please select at least one station to continue.')

    elif not selected_pollutants:
      st.warning('Please select at least one pollutant type.')
  
  with col2:
    st.subheader('Weekday vs. Weekend') #st.markdown('#### Diurnal Pattern Weekday vs. Weekend')
    # Kondisi Awal: Pilih Satu Hari
    if start_date == end_date:
      st.info('Visualisasi Weekday vs Weekend tidak tersedia untuk 1 hari.')
    elif selected_stations and selected_pollutants:
      filtered_df = main_df[main_df['station'].isin(selected_stations)].copy()
      filtered_df['datetime'] = pd.to_datetime(filtered_df[['year', 'month', 'day', 'hour']])
      filtered_df['hour'] = filtered_df['datetime'].dt.hour
      filtered_df['dayofweek'] = filtered_df['datetime'].dt.dayofweek
      filtered_df['weekend'] = filtered_df['dayofweek'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

      all_pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
      selected_pollutants_set = set(selected_pollutants)
      all_pollutants_set = set(all_pollutants)

      fig, ax = plt.subplots(figsize=(10,6))

      # Kondisi 1: Pilih Semua Polutan
      if selected_pollutants_set == all_pollutants_set:
        filtered_df['total_pollutant'] = filtered_df[all_pollutants].sum(axis=1)
        diurnal_week = filtered_df.groupby(['hour', 'weekend'])['total_pollutant'].mean().reset_index()
        sns.lineplot(data=diurnal_week, x='hour', y='total_pollutant', hue='weekend', marker='o', ax=ax)
        ax.set_title('Weekday vs Weekend: Diurnal Pattern of All Pollutant Concentration', fontsize=18)

      # Kondisi 2: Pilih Beberapa Polutan
      elif len(selected_pollutants) > 1:
        filtered_df['total_pollutant'] = filtered_df[selected_pollutants].sum(axis=1)
        diurnal_week = filtered_df.groupby(['hour', 'weekend'])['total_pollutant'].mean().reset_index()
        sns.lineplot(data=diurnal_week, x='hour', y='total_pollutant', hue='weekend', marker='o', ax=ax)
        ax.set_title('Weekday vs Weekend: Diurnal Pattern of Selected Pollutant Concentration', fontsize=18)

        # Kondisi 3: Pilih Satu Polutan
      else:
        pollutant = selected_pollutants[0]
        diurnal_week = filtered_df.groupby(['hour', 'weekend'])[pollutant].mean().reset_index()
        sns.lineplot(data=diurnal_week, x='hour', y=pollutant, hue='weekend', marker='o', ax=ax)
        ax.set_title(f'Weekday vs Weekend: Diurnal Pattern of {pollutant} Concentration', fontsize=18)

      ax.set_xlabel('Hour of Day', fontsize=16)
      ax.set_ylabel('Total Average Concentration (µg/m³)', fontsize=16)
      ax.set_xticks(range(0, 24))
      ax.tick_params(axis='both', labelsize=14)
      ax.grid(True)
      ax.legend(title='Day Type', fontsize=14)
      
      plt.tight_layout()
      st.pyplot(fig)
    
      # Heatmap: Jam vs Nama Hari
      with st.expander('View Heatmap: Hour vs Day Name'):
        if 'total_pollutant' not in filtered_df.columns:
          if selected_pollutants_set == all_pollutants_set:
            filtered_df['total_pollutant'] = filtered_df[all_pollutants].sum(axis=1)
          elif len(selected_pollutants) > 1:
            filtered_df['total_pollutant'] = filtered_df[selected_pollutants].sum(axis=1)
          else:
            pollutant = selected_pollutants[0]
            filtered_df['total_pollutant'] = filtered_df[pollutant]

        filtered_df['day_name'] = filtered_df['datetime'].dt.day_name()

        ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        heatmap_data = (
          filtered_df
          .groupby(['day_name', 'hour'])['total_pollutant']
          .mean()
          .unstack()
          .reindex(ordered_days)
        )

        fig_hm, ax_hm = plt.subplots(figsize=(12, 6))
        sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax_hm, linewidths=0.5, linecolor='white')
        ax_hm.set_title('Heatmap: Total Average Concentration per Hour and Day Name', fontsize=18)
        ax_hm.set_xlabel('Hour of Day', fontsize=16)
        ax_hm.set_ylabel('Day Name', fontsize=16)
        ax.tick_params(axis='both', labelsize=14)

        plt.tight_layout()
        st.pyplot(fig_hm)
        
    elif not selected_stations:
      st.warning('Please select at least one station to continue.')

    elif not selected_pollutants:
      st.warning('Please select at least one pollutant type.')
