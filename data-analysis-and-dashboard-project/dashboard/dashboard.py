import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

all_df = pd.read_csv('dashboard/all_data.csv')

# Menyortir dan memastikan kolom datetime
datetime_columns = ['datetime']
all_df.sort_values(by='datetime', inplace=True)
all_df.reset_index(inplace=True)

for column in datetime_columns:
  all_df[column] = pd.to_datetime(all_df[column])

min_date = all_df['datetime'].min()
max_date = all_df['datetime'].max()

st.set_page_config(layout='wide')

# Filter Tanggal, Statiun, Polutan, dan Periode di Sidebar
with st.sidebar:
  st.sidebar.header('Date')
  start_date = st.date_input('Start Date', value=min_date)
  end_date = st.date_input('End Date', value=max_date)

  st.sidebar.header('Station')
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

  st.sidebar.header('Pollutant')
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

  st.sidebar.header('Period')
  selected_period = st.selectbox(
    label = 'Choose Period',
    options = ['Daily', 'Weekly', 'Monthly'],
    index = 0,
    label_visibility = 'collapsed'
  )
  resample_map = {
    'Daily': 'D',
    'Weekly': 'W',
    'Monthly': 'M'
  }
  resample_freq = resample_map[selected_period]

if start_date > end_date:
  st.error('Tanggal Mulai tidak boleh lebih besar dari Tanggal Selesai!')

main_df = all_df[(all_df['datetime'] >= str(start_date)) & 
                 (all_df['datetime'] <= str(end_date))]

st.header('Air Quality Dashboard :fog:')

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
  #daily_aqi_cn = filtered_df['AQI_CN'].resample('D').max()

  with col1:
    st.subheader('Tren AQI') #st.markdown('#### Tren AQI')
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(aqi_resampled.index, aqi_resampled, color='red', linewidth=1.5)
    ax.set_title(f'Tren AQI - Standar China ({selected_day_type})')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Nilai AQI')
    ax.grid(True)
    st. pyplot(fig)
    
  with col2:
    st.subheader('Sebaran Kategori AQI') #st.markdown('#### Sebaran Kategori AQI')
    aqi_categories = [
      (0, 50, 'Excellent'),
      (51, 100, 'Good'),
      (101, 150, 'Slight Pollution'),
      (151, 200, 'Moderate Pollution'),
      (201, 300, 'Heavy Pollution'),
      (301, 500, 'Severe Pollution')
    ]
    kategori_counts = []
    for low, high, label in aqi_categories:
      count = filtered_df[(filtered_df['AQI_CN'] >= low) & (filtered_df['AQI_CN'] <= high)].shape[0]
      kategori_counts.append(count)

    kategori_labels = [cat for _, _, cat in aqi_categories]
    colors = ['green', 'yellow', 'orange', 'red', 'purple', 'maroon']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(kategori_labels, kategori_counts, color=colors)
    ax.set_title('Sebaran Kategori AQI Berdasarkan Data Terfilter')
    ax.set_xlabel('Kategori AQI')
    ax.set_ylabel('Jumlah Data')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for idx, bar in enumerate(bars):
      height = bar.get_height()
      ax.text(bar.get_x() + bar.get_width()/2, height + 3, str(height),
              ha='center', va='bottom', fontsize=9, fontweight='bold')

    st.pyplot(fig)

with tab2:
  # st.subheader(':chart_with_upwards_trend: Pollutant Trends')
  col1, col2 = st.columns(2)
  with col1:
    st.subheader('Demografi Konsentrasi Polutan') #st.markdown('#### Demografi Konsentrasi Polutan')
    if selected_stations and selected_pollutants:
      filtered_df = main_df[main_df['station'].isin(selected_stations)].copy()
      pollutants_agg_df = filtered_df.groupby(by='station')[selected_pollutants].mean()
      pollutants_agg_df['Total Rata-Rata Polutan'] = pollutants_agg_df.sum(axis=1)

      max_station = pollutants_agg_df['Total Rata-Rata Polutan'].idxmax()
      min_station = pollutants_agg_df['Total Rata-Rata Polutan'].idxmin()
      
      colors = []
      for station in pollutants_agg_df.index:
        if station == max_station:
          colors.append('red')
        elif station == min_station:
          colors.append('green')
        else:
          colors.append('lightgray')

      fig, ax = plt.subplots(figsize=(10,6))
      pollutants_agg_df['Total Rata-Rata Polutan'].plot(kind='bar', color=colors, ax=ax)

      ax.set_title('Total Rata-Rata Polutan per Stasiun (Terfilter)')
      ax.set_xlabel('Stasiun')
      ax.set_ylabel('Jumlah Rata-Rata Konsentrasi Polutan')
      ax.grid(axis='y', linestyle='--', alpha=0.7)
      plt.xticks(rotation=45)
      plt.tight_layout()
      st.pyplot(fig)

      df_to_display = pollutants_agg_df.drop(columns=['Total Rata-Rata Polutan'])
      with st.expander('Lihat Data Rata-Rata per Stasiun'):
        st.dataframe(df_to_display)
    
    elif not selected_stations:
      st.warning('Pilih setidaknya satu stasiun.')

    elif not selected_pollutants:
      st.warning('Pilih setidaknya satu jenis polutan.')

  with col2:
    st.subheader('Tren Konsentrasi Polutan') #st.markdown('#### Tren Konsentrasi Polutan')
    if selected_stations and selected_pollutants:      
      filtered_df = main_df[main_df['station'].isin(selected_stations)].copy()
      filtered_df['datetime'] = pd.to_datetime(filtered_df[['year', 'month', 'day', 'hour']])
      filtered_df.set_index('datetime', inplace=True)

      pollutant_resampled_df = filtered_df[selected_pollutants].resample(resample_freq).mean()
      pollutant_resampled_df['Total Polutan'] = pollutant_resampled_df.sum(axis=1)

      fig, ax = plt.subplots(figsize=(10,6))

      all_pollutants_set = set(['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'])
      selected_pollutants_set = set(selected_pollutants)

      if selected_pollutants_set == all_pollutants_set:
        ax.plot(
          pollutant_resampled_df.index, 
          pollutant_resampled_df['Total Polutan'], 
          label='Total Semua Polutan', 
          color='black', 
          linewidth=2.5)
      else:
        label = 'Total dari Polutan Terpilih' if len(selected_pollutants) > 1 else f'Tren {selected_pollutants[0]}'
        ax.plot(
          pollutant_resampled_df.index, 
          pollutant_resampled_df['Total Polutan'], 
          label=label, 
          color='blue', 
          linewidth=2.5)

      ax.set_title('Tren Rata-Rata Harian Konsentrasi Polutan')
      ax.set_xlabel('Tanggal')
      ax.set_ylabel('Konsentrasi Rata-rata')
      ax.legend()
      ax.grid(True, linestyle='--', alpha=0.5)
      plt.tight_layout()
      st.pyplot(fig)
    elif not selected_stations:
      st.warning('Pilih setidaknya satu stasiun.')

    elif not selected_pollutants:
      st.warning('Pilih setidaknya satu jenis polutan.')

with tab3:
  # st.subheader(':clock3: Diurnal Patterns')
  col1, col2 = st.columns(2)
  with col1:
    st.subheader('Diurnal Pattern All Day') #st.markdown('#### Diurnal Pattern All Day')
  with col2:
    st.subheader('Diurnal Pattern Weekday vs. Weekend') #st.markdown('#### Diurnal Pattern Weekday vs. Weekend')
