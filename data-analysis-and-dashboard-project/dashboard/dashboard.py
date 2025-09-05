import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
import emoji

sns.set_theme(style='dark')

base_dir = os.path.dirname(__file__)
zip_path = os.path.join(base_dir, 'all_data.zip')

all_df = pd.read_csv(zip_path, compression='zip')

# Sort datetime columns
datetime_columns = ['datetime']
all_df.sort_values(by = 'datetime', inplace=True)
all_df.reset_index(drop=True, inplace=True)

for column in datetime_columns:
  all_df[column] = pd.to_datetime(all_df[column])

min_date = all_df['datetime'].min().date()
max_date = all_df['datetime'].max().date()

st.set_page_config(layout='wide')

# Date, Station, Pollutant, and Periode Filter in Sidebar
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
  selected_stations = stations if 'All Stations' in choose_stations else choose_stations

  st.sidebar.header(':fog: Pollutant')
  pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
  choose_pollutants = st.sidebar.multiselect(
    label = 'Choose Pollutants',
    options = ['All Pollutants'] + pollutants,
    default = ['All Pollutants'],
    label_visibility = 'collapsed'
  )
  selected_pollutants = pollutants if 'All Pollutants' in choose_pollutants else choose_pollutants

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
  st.error("Oops! Start Date can't be later than the End Date.")

main_df = all_df[(all_df['datetime'].dt.date >= start_date) &
                 (all_df['datetime'].dt.date <= end_date)]

invalid_station = not selected_stations
invalid_pollutant = not selected_pollutants

if invalid_station:
  st.warning(':warning: Please select at least one station to continue.')
if invalid_pollutant:
  st.warning(':warning: Please select at least one pollutant type.')

if not invalid_station and not invalid_pollutant:
  filtered_df = main_df[main_df['station'].isin(selected_stations)].copy()
  filtered_df['datetime'] = pd.to_datetime(filtered_df[['year', 'month', 'day', 'hour']])
  filtered_df.set_index('datetime', inplace=True)
else:
  filtered_df = pd.DataFrame()
  
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
  col1, col2 = st.columns(2)
  
  with col1:
    st.subheader('AQI Trend')
    if not filtered_df.empty:
      aqi_resampled = filtered_df['AQI_CN'].resample(resample_freq).max()

      if selected_period == 'Hourly' and start_date == end_date:
        title_label = f'{selected_period} AQI Trend - {start_date}'
      else:
        title_label = f'{selected_period} AQI Trend - {start_date} to {end_date}'
      
      fig = go.Figure()
      fig.add_trace(go.Scatter(
        x = aqi_resampled.index,
        y = aqi_resampled.values,
        mode = 'lines',
        line = dict(color='royalblue', width=2),
        name = 'AQI'
      ))

      fig.update_layout(
        title = title_label,
        xaxis_title = 'Date and Time',
        yaxis_title = 'AQI Value',
        template = 'plotly_white',
        margin = dict(l=20, r=20, t=50, b=20),
        height = 400,
        font = dict(size=14),
      )

      st.plotly_chart(fig, use_container_width=True)
        
  with col2:
    st.subheader('AQI Category Distribution')
    if not filtered_df.empty:
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
      kategori_tooltips = []
      for low, high, label in aqi_categories:
        count = aqi_resampled[(aqi_resampled >= low) & (aqi_resampled <= high)].shape[0]
        kategori_counts.append(count)
        kategori_labels.append(f'{low}–{high}')
        kategori_tooltips.append(label)

      colors = ['green', 'yellow', 'orange', 'red', 'purple', 'maroon']

      fig = go.Figure(data=[
        go.Bar(
          x = kategori_labels,
          y = kategori_counts,
          marker_color = colors,
          text = kategori_counts,
          textposition = 'outside',
          hovertext = kategori_tooltips,
          hoverinfo = 'text+y'
        )
      ])

      fig.update_layout(
        title = f'AQI Category Distribution - {selected_period}',
        xaxis_title = 'AQI Value Range',
        yaxis_title = 'Data Count',
        template = 'plotly_white',
        margin = dict(l=20, r=20, t=50, b=20),
        height = 400,
        font = dict(size=14),
        showlegend = False
      )

      st.plotly_chart(fig, use_container_width=True)

with tab2:
  col1, col2 = st.columns(2)

  with col1:
    st.subheader('Demographics by Station')
    if not filtered_df.empty:
      pollutants_agg_df = filtered_df.groupby(by = 'station')[selected_pollutants].mean()
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

      fig = go.Figure(data=[
        go.Bar(
          x = pollutants_agg_df.index,
          y = pollutants_agg_df['total_average'],
          marker_color = colors,
          text = pollutants_agg_df['total_average'].round(2),
          textposition = 'outside',
          hovertemplate = '<b>%{x}</b><br>Total Avg: %{y:.2f} µg/m³<extra></extra>'
        )
      ])

      fig.update_layout(
        title = 'Average Pollutant Levels by Station',
        xaxis_title = 'Station',
        yaxis_title = 'Total Average Concentration (µg/m³)',
        template = 'plotly_white',
        margin = dict(l=20, r=20, t=50, b=20),
        height = 400,
        font = dict(size=14)
      )

      st.plotly_chart(fig, use_container_width=True)

      # Tabel data detail tanpa kolom total
      df_to_display = pollutants_agg_df.drop(columns=['total_average'])
      with st.expander('View Total Average Data per Station'):
        st.dataframe(df_to_display)
    
  with col2:
    st.subheader('Pollutant Concentration Trend')
    if not filtered_df.empty:
      pollutant_resampled_df = filtered_df[selected_pollutants].resample(resample_freq).mean()
      
      if selected_period == 'Hourly' and start_date == end_date:
        title_label = f'Pollutant Concentration Trend - {selected_period}, {start_date}'
      else:
        title_label = f'Pollutant Concentration Trend - {selected_period}, {start_date} to {end_date}'

      pollutant_resampled_df['total_average'] = pollutant_resampled_df.sum(axis=1)

      fig = go.Figure()

      all_pollutants_set = set(['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'])
      selected_pollutants_set = set(selected_pollutants)

      if selected_pollutants_set == all_pollutants_set:
        pollutant_resampled_df['total_average'] = pollutant_resampled_df.sum(axis=1)
        fig.add_trace(go.Scatter(
          x = pollutant_resampled_df.index,
          y = pollutant_resampled_df['total_average'],
          mode = 'lines',
          name = 'All Pollutants',
          line = dict(color='darkred', width=3),
          hovertemplate = '%{x}<br>Total Avg: %{y:.2f} µg/m³<extra></extra>'
        ))
      elif len(selected_pollutants) > 1:
        # Choose Selected Pollutants
        pollutant_resampled_df['total_average'] = pollutant_resampled_df.sum(axis=1)
        fig.add_trace(go.Scatter(
          x = pollutant_resampled_df.index,
          y = pollutant_resampled_df['total_average'],
          mode = 'lines',
          name = 'Selected Pollutants',
          line = dict(color='crimson', width=2.5),
          hovertemplate = '%{x}<br>Total Avg: %{y:.2f} µg/m³<extra></extra>'
        ))
      else:
        # Choose One Pollutant
        pollutant = selected_pollutants[0]
        fig.add_trace(go.Scatter(
          x = pollutant_resampled_df.index,
          y = pollutant_resampled_df[pollutant],
          mode = 'lines',
          name = pollutant,
          line = dict(color='darkred', width=2.5),
          hovertemplate = '%{x}<br>' + pollutant + ': %{y:.2f} µg/m³<extra></extra>'
        ))

      fig.update_layout(
        title = title_label,
        xaxis_title = 'Date and Time',
        yaxis_title = 'Total Average Concentration (µg/m³)',
        template = 'plotly_white',
        font = dict(size=14),
        margin = dict(l=20, r=20, t=50, b=20),
        height = 400,
        legend = dict(font=dict(size=12))
      )

      st.plotly_chart(fig, use_container_width=True)

with tab3:
  col1, col2 = st.columns(2)

  with col1:
    st.subheader('All Day')
    if not filtered_df.empty:
      all_pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
      selected_pollutants_set = set(selected_pollutants)
      all_pollutants_set = set(all_pollutants)

      df_diurnal = filtered_df.copy()
      df_diurnal['datetime'] = pd.to_datetime(df_diurnal[['year', 'month', 'day', 'hour']])

      # Condition 1: Choose All Pollutants
      if selected_pollutants_set == all_pollutants_set:
        pollutant_label = 'Total Pollutants'
        df_diurnal['total_pollutant'] = df_diurnal[all_pollutants].sum(axis=1)
        
      # Condition 2: Choose Selected Pollutants
      elif len(selected_pollutants) > 1:
        pollutant_label = 'Selected Pollutants'
        df_diurnal['total_pollutant'] = df_diurnal[selected_pollutants].sum(axis=1)
        
      # Condition 3: Choose One Pollutant
      else:
        pollutant = selected_pollutants[0]
        pollutant_label = pollutant
        df_diurnal['total_pollutant'] = df_diurnal[pollutant]
      
      if start_date == end_date:
        title = f'Diurnal Profile of {pollutant_label} – {start_date}'
      else:
        title = f'Diurnal Profile of {pollutant_label} – {start_date} to {end_date}'
      
      y_data = df_diurnal.groupby('hour')['total_pollutant'].mean()
      
      fig = go.Figure()
      fig.add_trace(go.Scatter(
        x = y_data.index,
        y = y_data.values,
        mode = 'lines+markers',
        line = dict(color='steelblue'),
        marker = dict(size=8),
        name = 'Average Concentration'
      ))
      
      fig.update_layout(
        title = title,
        xaxis_title = 'Hour of Day',
        yaxis_title = 'Total Average Concentration (µg/m³)',
        xaxis = dict(tickmode='linear', tick0=0, dtick=1, range=[0,23], gridcolor='LightGray'),
        yaxis = dict(gridcolor='LightGray'),
        template = 'plotly_white',
        font = dict(size=14),
        height = 400,
        margin = dict(l=40, r=40, t=60, b=40)
      )

      st.plotly_chart(fig, use_container_width=True)

      # Heatmap: Hour vs Day of the Month
      with st.expander('View Heatmap: Hour vs Day of the Month'):
        df_heat = df_diurnal.copy()
        df_heat['day'] = df_heat['datetime'].dt.day

        heatmap_data = (
          df_heat
          .groupby(['day', 'hour'])['total_pollutant']
          .mean()
          .unstack()
        )

        fig_hm = px.imshow(
          heatmap_data,
          labels = dict(x='Hour of Day', y='Day of the Month'), #color='Avg Concentration (µg/m³)
          x = heatmap_data.columns,
          y = heatmap_data.index,
          color_continuous_scale = 'YlOrRd',
          aspect = 'auto',
        )
        
        fig_hm.update_layout(
          title = 'Heatmap: Hourly Profile of Total Pollutants by Day of the Month',
          height = 250,
          font = dict(size=14),
          margin = dict(l=40, r=40, t=50, b=40))
        
        st.plotly_chart(fig_hm, use_container_width=True)

  with col2:
    st.subheader('Weekday vs. Weekend')
    if start_date == end_date:
      st.info('Weekday vs Weekend visualization is not available for a single day.')
    
    elif not invalid_station and not invalid_pollutant:
      df_work = main_df[main_df['station'].isin(selected_stations)].copy()
      df_work['datetime'] = pd.to_datetime(df_work[['year', 'month', 'day', 'hour']])
      df_work['hour'] = df_work['datetime'].dt.hour
      df_work['dayofweek'] = df_work['datetime'].dt.dayofweek
      df_work['weekend'] = df_work['dayofweek'].apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

      all_pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
      selected_pollutants_set = set(selected_pollutants)
      all_pollutants_set = set(all_pollutants)

      # Condition 1: Choose All Pollutants
      if selected_pollutants_set == all_pollutants_set:
        pollutant_label = 'Total Pollutants'
        df_work['total_pollutant'] = df_work[all_pollutants].sum(axis=1)
        y_col = 'total_pollutant'
        diurnal_week = df_work.groupby(['hour', 'weekend'])['total_pollutant'].mean().reset_index()

      # Condition 2: Choose Selected Pollutants
      elif len(selected_pollutants) > 1:
        pollutant_label = 'Selected Pollutants'
        df_work['total_pollutant'] = df_work[selected_pollutants].sum(axis=1)
        y_col = 'total_pollutant'
        diurnal_week = df_work.groupby(['hour', 'weekend'])['total_pollutant'].mean().reset_index()

      # Condition 3: Choose One Pollutant
      else:
        pollutant = selected_pollutants[0]
        pollutant_label = pollutant
        y_col = pollutant
        diurnal_week = df_work.groupby(['hour', 'weekend'])[pollutant].mean().reset_index()
      
      title = f'Weekday vs Weekend: Diurnal Profile of {pollutant_label} - {start_date} to {end_date}'
        
      fig = go.Figure()
      for day_type, group_df in diurnal_week.groupby('weekend'):
        if day_type == 'Weekday':
          color = 'royalblue'
        elif day_type == 'Weekend':
          color = 'orange'

        fig.add_trace(go.Scatter(
          x = group_df['hour'],
          y = group_df[y_col],
          mode = 'lines+markers',
          name = day_type,
          marker = dict(size=8),
          line=dict(color=color)
      ))

      fig.update_layout(
        title = title,
        xaxis_title = 'Hour of Day',
        yaxis_title = 'Total Average Concentration (µg/m³)',
        xaxis = dict(tickmode='linear', tick0=0, dtick=1, range=[0,23], gridcolor='LightGray'),
        yaxis = dict(gridcolor='LightGray'),
        template = 'plotly_white',
        font = dict(size=14),
        height = 400,
        margin = dict(l=40, r=40, t=60, b=40),
        legend_title_text = 'Day Type',
        legend = dict(
          x = 0.98,
          y = 0.98,
          xanchor = 'right',
          yanchor = 'top',
          bgcolor = 'rgba(255,255,255,0.5)',
          bordercolor = 'lightgray',
          borderwidth = 1
        )
      )
            
      st.plotly_chart(fig, use_container_width=True)
    
      # Heatmap: Hour vs Day Name
      with st.expander('View Heatmap: Hour vs Day Name'):
        df_heat = df_work.copy()
        
        if 'total_pollutant' not in df_heat.columns:
          if selected_pollutants_set == all_pollutants_set:
            df_heat['total_pollutant'] = df_heat[all_pollutants].sum(axis=1)
          elif len(selected_pollutants) > 1:
            df_heat['total_pollutant'] = df_heat[selected_pollutants].sum(axis=1)
          else:
            pollutant = selected_pollutants[0]
            df_heat['total_pollutant'] = df_heat[pollutant]

        df_heat['day_name'] = df_heat['datetime'].dt.day_name()
        ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Onerday', 'Sunday']
        
        heatmap_data = (
          df_heat
          .groupby(['day_name', 'hour'])['total_pollutant']
          .mean()
          .unstack()
          .reindex(ordered_days)
        )

        fig_hm = px.imshow(
          heatmap_data,
          labels = dict(x='Hour of Day', y='Day Name'), #color='Avg Concentration (µg/m³)
          x = heatmap_data.columns,
          y = heatmap_data.index,
          color_continuous_scale = 'YlOrRd',
          aspect = 'auto',
        )
        
        fig_hm.update_layout(
          title = 'Heatmap: Hourly Profile of Total Pollutants by Day of Name',
          height = 250,
          font = dict(size=14),
          margin = dict(l=40, r=40, t=50, b=40)
        )

        st.plotly_chart(fig_hm, use_container_width=True)
    
