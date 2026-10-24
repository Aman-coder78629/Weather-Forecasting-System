# ================================================================
# DELHI WEATHER FORECASTING SYSTEM
# Interactive Streamlit Dashboard + Machine Learning
# ================================================================
# Install: python -m pip install streamlit pandas numpy plotly scikit-learn
# Run:     python -m streamlit run weather_forecasting_system.py
# ================================================================
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import timedelta
from io import BytesIO
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title='Delhi Weather Intelligence', page_icon='🌦️', layout='wide', initial_sidebar_state='expanded')
DEFAULT_DATA_PATH = r'C:\Users\amans\Downloads\DailyDelhiClimateTest.csv'
DATE_CANDIDATES = ['date', 'datetime', 'timestamp', 'day']
COLORS = ['#38bdf8','#818cf8','#c084fc','#34d399','#fbbf24','#fb7185','#2dd4bf','#a3e635']

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif}.stApp{background:radial-gradient(circle at top left,rgba(49,130,206,.13),transparent 32%),linear-gradient(135deg,#0b1220,#101b2d 50%,#111827);color:#f8fafc}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#101827,#172338);border-right:1px solid rgba(148,163,184,.15)}
[data-testid="stMetric"]{background:linear-gradient(145deg,rgba(30,41,59,.92),rgba(15,23,42,.96));border:1px solid rgba(125,211,252,.2);padding:18px;border-radius:18px;box-shadow:0 8px 28px rgba(0,0,0,.18)}
[data-testid="stMetricLabel"]{color:#cbd5e1;font-weight:600}[data-testid="stMetricValue"]{color:#7dd3fc;font-weight:800}
.hero{padding:30px;border-radius:24px;margin-bottom:24px;background:linear-gradient(115deg,rgba(14,165,233,.25),rgba(99,102,241,.22),rgba(168,85,247,.16));border:1px solid rgba(125,211,252,.3)}
.hero h1{font-size:clamp(28px,4vw,44px);color:#f8fafc;margin:0;font-weight:800}.hero p{color:#cbd5e1;font-size:16px;margin-top:12px;margin-bottom:0}
.section-heading{color:#e2e8f0;font-weight:800;font-size:23px;padding-bottom:8px;border-bottom:1px solid rgba(148,163,184,.2);margin:12px 0 18px}
.small-note{color:#94a3b8;font-size:12px}.stDownloadButton button,div.stButton>button{border-radius:10px;font-weight:700}
</style>''', unsafe_allow_html=True)

def norm(name):
    return str(name).strip().lower().replace(' ','_').replace('-','_').replace('/','_')

def find_col(columns, candidates):
    lookup={norm(c):c for c in columns}
    return next((lookup[c] for c in candidates if c in lookup),None)

def fmt(v):
    return 'N/A' if pd.isna(v) else f'{float(v):,.2f}'

def style(fig,height=390):
    fig.update_layout(template='plotly_dark',height=height,margin=dict(l=18,r=18,t=55,b=20),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(15,23,42,.3)',font=dict(family='Inter,sans-serif',color='#e2e8f0',size=12),title_font=dict(size=17,color='#f8fafc'))
    fig.update_xaxes(gridcolor='rgba(148,163,184,.12)'); fig.update_yaxes(gridcolor='rgba(148,163,184,.12)')
    return fig

@st.cache_data(show_spinner=False)
def read_uploaded(data_bytes): return pd.read_csv(BytesIO(data_bytes))
@st.cache_data(show_spinner=False)
def read_path(path,mtime):
    del mtime
    return pd.read_csv(path)

def load_data(uploaded,path):
    try:
        if uploaded is not None: return read_uploaded(uploaded.getvalue()),f'Uploaded file: {uploaded.name}'
        p=Path(path)
        if p.exists(): return read_path(str(p),p.stat().st_mtime),str(p)
        return None,None
    except Exception as exc:
        st.error(f'Could not read CSV: {exc}'); return None,None

def clean_data(raw):
    df=raw.copy(); df.columns=[norm(c) for c in df.columns]; original=len(df)
    df=df.drop_duplicates().reset_index(drop=True)
    date_col=find_col(df.columns,DATE_CANDIDATES)
    if date_col:
        df[date_col]=pd.to_datetime(df[date_col],errors='coerce'); df=df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    for c in df.columns:
        if c==date_col: continue
        if df[c].dtype==object:
            converted=pd.to_numeric(df[c],errors='coerce'); nonmissing=df[c].notna().sum()
            if nonmissing and converted.notna().sum()/nonmissing>=.75: df[c]=converted
        if pd.api.types.is_numeric_dtype(df[c]): df[c]=df[c].replace([np.inf,-np.inf],np.nan)
    for c in df.select_dtypes(include=np.number).columns:
        df[c]=df[c].interpolate(limit_direction='both')
        if df[c].isna().any():
            median=df[c].median(); df[c]=df[c].fillna(0 if pd.isna(median) else median)
    return df,{'original_rows':original,'cleaned_rows':len(df),'duplicates_removed':original-len(df),'missing_values_remaining':int(df.isna().sum().sum()),'date_column':date_col}

def lag_features(series,lags=7):
    frame=pd.DataFrame({'target':pd.to_numeric(series,errors='coerce')})
    for i in range(1,lags+1): frame[f'lag_{i}']=frame.target.shift(i)
    frame['rolling_mean_3']=frame.target.shift(1).rolling(3).mean(); frame['rolling_mean_7']=frame.target.shift(1).rolling(7).mean(); frame['rolling_std_7']=frame.target.shift(1).rolling(7).std()
    return frame.replace([np.inf,-np.inf],np.nan).dropna().reset_index(drop=True)

def models():
    return {'Linear Regression':LinearRegression(),'Ridge Regression':make_pipeline(StandardScaler(),Ridge(alpha=1.0)),'Random Forest':RandomForestRegressor(n_estimators=150,random_state=42,min_samples_leaf=2,n_jobs=-1),'Extra Trees':ExtraTreesRegressor(n_estimators=150,random_state=42,min_samples_leaf=2,n_jobs=-1),'Gradient Boosting':GradientBoostingRegressor(n_estimators=100,learning_rate=.05,max_depth=2,random_state=42),'Histogram Gradient Boosting':HistGradientBoostingRegressor(max_iter=100,learning_rate=.05,random_state=42)}

def evaluate(series,test_fraction=.2):
    frame=lag_features(series)
    if len(frame)<15: return None,None,'Not enough observations to evaluate models reliably. Use a longer historical dataset (at least about 25 observations is recommended).'
    X=frame.drop(columns='target'); y=frame.target; split=max(10,min(int(len(frame)*(1-test_fraction)),len(frame)-3))
    Xtr,Xte=X.iloc[:split],X.iloc[split:]; ytr,yte=y.iloc[:split],y.iloc[split:]
    results=[]; preds={}; fitted={}
    for name,model in models().items():
        try:
            model.fit(Xtr,ytr); p=model.predict(Xte)
            results.append({'Model':name,'MAE':mean_absolute_error(yte,p),'RMSE':np.sqrt(mean_squared_error(yte,p)),'R2':r2_score(yte,p) if len(yte)>1 else np.nan})
            preds[name]=pd.DataFrame({'Actual':yte.to_numpy(),'Predicted':p}); fitted[name]=model
        except Exception as exc: st.warning(f'{name} failed: {exc}')
    if not results: return None,None,'All selected models failed.'
    return pd.DataFrame(results).sort_values('MAE').reset_index(drop=True),{'models':fitted,'predictions':preds},None

def forecast(series,model,horizon):
    values=pd.to_numeric(series,errors='coerce').dropna().astype(float).tolist()
    if len(values)<7: raise ValueError('At least 7 valid observations are required.')
    out=[]; history=values.copy()
    for _ in range(horizon):
        row={f'lag_{i}':history[-i] for i in range(1,8)}
        row.update(rolling_mean_3=float(np.mean(history[-3:])),rolling_mean_7=float(np.mean(history[-7:])),rolling_std_7=float(np.std(history[-7:],ddof=1)))
        value=float(model.predict(pd.DataFrame([row]))[0]); out.append(value); history.append(value)
    return out

# Sidebar
st.sidebar.title('🌦️ Weather Intelligence'); st.sidebar.caption('Delhi Weather Forecasting System')
upload=st.sidebar.file_uploader('Upload a weather CSV',type=['csv']); local_path=st.sidebar.text_input('Local CSV path',value=DEFAULT_DATA_PATH)
st.sidebar.caption('Expected columns when available: date, meantemp, humidity, wind_speed, meanpressure')
st.markdown('<div class="hero"><h1>🌦️ Delhi Weather Intelligence</h1><p>Explore historical weather data, compare machine-learning models, and generate short-term estimates.</p></div>',unsafe_allow_html=True)
raw,source=load_data(upload,local_path)
if raw is None: st.warning('Dataset not found. Upload a CSV in the sidebar or update the local CSV path.'); st.info('Default path: C:\\Users\\amans\\Downloads\\DailyDelhiClimateTest.csv'); st.stop()
if raw.empty: st.error('The selected CSV is empty.'); st.stop()
data,report=clean_data(raw)
if data.empty: st.error('No usable records remain after cleaning. Check your CSV.'); st.stop()
numeric=data.select_dtypes(include=np.number).columns.tolist()
if not numeric: st.error('No numeric columns detected in the CSV.'); st.stop()
detected={'temperature':find_col(data.columns,['meantemp','temperature','temp','mean_temp']),'humidity':find_col(data.columns,['humidity','relative_humidity']),'wind':find_col(data.columns,['wind_speed','windspeed','wind']),'pressure':find_col(data.columns,['meanpressure','pressure','air_pressure']),'date':find_col(data.columns,DATE_CANDIDATES)}
default_target=detected['temperature'] if detected['temperature'] in numeric else numeric[0]
target=st.sidebar.selectbox('Forecast target / primary variable',numeric,index=numeric.index(default_target)); date_col=detected['date']
temp_col=detected['temperature'] if detected['temperature'] in numeric else target
humidity_col=detected['humidity'] if detected['humidity'] in numeric else None
wind_col=detected['wind'] if detected['wind'] in numeric else None
pressure_col=detected['pressure'] if detected['pressure'] in numeric else None
st.caption(f'**Data source:** {source} · **Rows:** {len(data):,} · **Columns:** {len(data.columns)}')
st.markdown('<div class="section-heading">📌 Weather Overview</div>',unsafe_allow_html=True)
k=st.columns(4); k[0].metric('Records',f'{len(data):,}'); k[1].metric(f'Average {target}',fmt(data[target].mean())); k[2].metric(f'Minimum {target}',fmt(data[target].min())); k[3].metric(f'Maximum {target}',fmt(data[target].max()))
if humidity_col:
    k=st.columns(3); k[0].metric('Average Humidity',fmt(data[humidity_col].mean())); k[1].metric('Duplicates Removed',f"{report['duplicates_removed']:,}"); k[2].metric('Missing Values Left',f"{report['missing_values_remaining']:,}")

def x_axis_frame(df):
    copy=df.copy()
    if date_col: return copy,date_col
    copy['Observation']=np.arange(1,len(copy)+1); return copy,'Observation'

def line_chart(column,title,color):
    frame,x=x_axis_frame(data); fig=px.line(frame,x=x,y=column,title=title,markers=True,color_discrete_sequence=[color]); fig.update_traces(line_width=3); return style(fig)

t1,t2,t3,t4,t5=st.tabs(['🏠 Overview','📊 Explore Data','🤖 Model Evaluation','🔮 Forecast','🧹 Data & Downloads'])
with t1:
    st.markdown('<div class="section-heading">Historical Weather at a Glance</div>',unsafe_allow_html=True)
    a,b=st.columns(2); a.plotly_chart(line_chart(temp_col,'🌡️ Temperature Trend',COLORS[0]),use_container_width=True)
    if humidity_col: b.plotly_chart(line_chart(humidity_col,'💧 Humidity Trend',COLORS[1]),use_container_width=True)
    else: b.plotly_chart(style(px.histogram(data,x=target,marginal='box',title=f'Distribution of {target}',color_discrete_sequence=[COLORS[2]])),use_container_width=True)
    a,b=st.columns(2)
    if wind_col: a.plotly_chart(style(px.area(x=data[date_col] if date_col else np.arange(len(data)),y=data[wind_col],title='💨 Wind Speed Over Time',labels={'x':'Date / observation','y':wind_col},color_discrete_sequence=[COLORS[3]])),use_container_width=True)
    else: a.plotly_chart(style(px.histogram(data,x=target,title='Weather Variable Distribution',color_discrete_sequence=[COLORS[3]])),use_container_width=True)
    if pressure_col: b.plotly_chart(line_chart(pressure_col,'🧭 Atmospheric Pressure',COLORS[4]),use_container_width=True)
    else: b.plotly_chart(style(px.imshow(data[numeric].corr(),text_auto='.2f',aspect='auto',color_continuous_scale='RdBu_r',zmin=-1,zmax=1,title='🔗 Variable Correlation')),use_container_width=True)
with t2:
    st.markdown('<div class="section-heading">Interactive Data Exploration</div>',unsafe_allow_html=True)
    c1,c2,c3,c4=st.tabs(['Distribution','Correlation','Monthly Summary','Scatter Analysis'])
    with c1:
        variable=st.selectbox('Variable for distribution',numeric); st.plotly_chart(style(px.histogram(data,x=variable,nbins=30,marginal='box',title=f'Distribution of {variable}',color_discrete_sequence=[COLORS[2]])),use_container_width=True)
    with c2: st.plotly_chart(style(px.imshow(data[numeric].corr(),text_auto='.2f',aspect='auto',color_continuous_scale='RdBu_r',zmin=-1,zmax=1,title='Weather Variable Correlation'),max(600,45*len(numeric)),),use_container_width=True)
    with c3:
        if date_col:
            monthly=data[[date_col,temp_col]].dropna().copy(); monthly['Month']=monthly[date_col].dt.month; monthly=monthly.groupby('Month',as_index=False)[temp_col].mean(); fig=px.bar(monthly,x='Month',y=temp_col,title='🗓️ Average Temperature by Month',color=temp_col,color_continuous_scale='Tealgrn'); fig.update_xaxes(dtick=1); st.plotly_chart(style(fig),use_container_width=True)
        else: st.info('Monthly summary requires a date column.')
    with c4:
        if len(numeric)>1:
            sx=st.selectbox('X-axis variable',numeric,index=0); sy=st.selectbox('Y-axis variable',numeric,index=1); fig=px.scatter(data,x=sx,y=sy,title=f'{sy} vs {sx}',trendline='ols',color_discrete_sequence=[COLORS[6]]); st.plotly_chart(style(fig),use_container_width=True)
        else: st.info('At least two numeric columns are needed.')
with t3:
    st.markdown('<div class="section-heading">Machine-Learning Model Comparison</div>',unsafe_allow_html=True)
    split=st.slider('Test data proportion',.10,.35,.20,.05)
    with st.spinner('Training and evaluating models...'): results,bundle,error=evaluate(data[target],split)
    if error: st.warning(error)
    else:
        m=st.columns(3); m[0].metric('Best model (lowest MAE)',str(results.iloc[0]['Model'])); m[1].metric('Best MAE',fmt(results.iloc[0]['MAE'])); m[2].metric('Best RMSE',fmt(results.iloc[0]['RMSE']))
        st.dataframe(results.style.format({'MAE':'{:.4f}','RMSE':'{:.4f}','R2':'{:.4f}'}),use_container_width=True)
        metric=st.selectbox('Compare models by metric',['MAE','RMSE','R2']); sorted_results=results.sort_values(metric,ascending=metric!='R2'); fig=px.bar(sorted_results,x='Model',y=metric,color='Model',title=f'Model Comparison by {metric}',color_discrete_sequence=COLORS); fig.update_layout(showlegend=False,xaxis_tickangle=-20); st.plotly_chart(style(fig),use_container_width=True)
        chosen=st.selectbox('View test predictions for',list(bundle['predictions'])); p=bundle['predictions'][chosen].copy(); p['Test observation']=np.arange(1,len(p)+1); fig=go.Figure(); fig.add_trace(go.Scatter(x=p['Test observation'],y=p['Actual'],mode='lines+markers',name='Actual')); fig.add_trace(go.Scatter(x=p['Test observation'],y=p['Predicted'],mode='lines+markers',name='Predicted',line=dict(dash='dash'))); st.plotly_chart(style(fig),use_container_width=True)
        st.download_button('⬇️ Download model evaluation results',results.to_csv(index=False).encode(), 'weather_model_evaluation.csv','text/csv')
with t4:
    st.markdown('<div class="section-heading">Short-Term Weather Forecast</div>',unsafe_allow_html=True)
    st.caption('Recursive model estimates based on historical data; these are not official weather forecasts or warnings.')
    horizon=st.slider('Forecast horizon (days / steps)',1,30,7); split2=st.slider('Evaluation split for forecast model',.10,.35,.20,.05,key='forecast_split')
    with st.spinner('Preparing forecast model...'): res2,bundle2,err2=evaluate(data[target],split2)
    if err2: st.warning(err2)
    else:
        model_name=st.selectbox('Forecast model',list(bundle2['models']),key='forecast_model')
        if st.button('🔮 Generate Forecast',type='primary'):
            try:
                full=lag_features(data[target]); model=models()[model_name]; model.fit(full.drop(columns='target'),full['target']); values=forecast(data[target],model,horizon)
                if date_col and data[date_col].notna().any():
                    last=data[date_col].dropna().max(); frame=pd.DataFrame({'Date':[last+timedelta(days=i) for i in range(1,horizon+1)],f'Forecast_{target}':values})
                else: frame=pd.DataFrame({'Observation':np.arange(len(data)+1,len(data)+horizon+1),f'Forecast_{target}':values})
                st.session_state['weather_forecast_frame']=frame; st.session_state['weather_forecast_model']=model_name
            except Exception as exc: st.error(f'Could not generate forecast: {exc}')
    if 'weather_forecast_frame' in st.session_state:
        frame=st.session_state['weather_forecast_frame']; st.subheader(f"Forecast output — {st.session_state.get('weather_forecast_model','Selected model')}"); st.dataframe(frame,use_container_width=True)
        fcol=next(c for c in frame if c.startswith('Forecast_')); fig=go.Figure()
        if date_col and 'Date' in frame:
            hist=data[[date_col,target]].dropna().tail(90); fig.add_trace(go.Scatter(x=hist[date_col],y=hist[target],mode='lines',name='Historical')); fig.add_trace(go.Scatter(x=frame['Date'],y=frame[fcol],mode='lines+markers',name='Forecast',line=dict(dash='dash')))
        else:
            hist=data[target].dropna().tail(90); start=len(data)-len(hist)+1; fig.add_trace(go.Scatter(x=np.arange(start,len(data)+1),y=hist,mode='lines',name='Historical')); fig.add_trace(go.Scatter(x=frame['Observation'],y=frame[fcol],mode='lines+markers',name='Forecast',line=dict(dash='dash')))
        fig.update_layout(title='Historical Values and Forecast',xaxis_title='Date / observation',yaxis_title=target); st.plotly_chart(style(fig),use_container_width=True)
        st.download_button('⬇️ Download forecast CSV',frame.to_csv(index=False).encode(),'delhi_weather_forecast.csv','text/csv')
with t5:
    st.markdown('<div class="section-heading">Data Quality and Downloads</div>',unsafe_allow_html=True)
    q=st.columns(4); q[0].metric('Original rows',f"{report['original_rows']:,}"); q[1].metric('Cleaned rows',f"{report['cleaned_rows']:,}"); q[2].metric('Duplicates removed',f"{report['duplicates_removed']:,}"); q[3].metric('Missing values left',f"{report['missing_values_remaining']:,}")
    st.subheader('Cleaned dataset preview'); st.dataframe(data.head(100),use_container_width=True)
    missing=data.isna().sum().rename('Missing values').reset_index().rename(columns={'index':'Column'}); st.subheader('Missing values by column'); st.dataframe(missing,use_container_width=True)
    st.download_button('⬇️ Download cleaned dataset',data.to_csv(index=False).encode(),'cleaned_delhi_weather_data.csv','text/csv')
    info=pd.DataFrame({'Column':data.columns,'Data type':[str(x) for x in data.dtypes],'Non-null values':[int(data[c].notna().sum()) for c in data.columns],'Missing values':[int(data[c].isna().sum()) for c in data.columns]}); st.subheader('Dataset information'); st.dataframe(info,use_container_width=True)
st.markdown('---'); st.markdown('<p class="small-note">Delhi Weather Intelligence · Streamlit + Plotly + scikit-learn. Forecasts are educational estimates and should not be used as official weather warnings.</p>',unsafe_allow_html=True)
