from pathlib import Path
import pandas as pd
import json
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'observed' / 'raw'
OUT = ROOT / 'data' / 'observed' / 'processed'
OUT.mkdir(parents=True, exist_ok=True)

def load(filename):
    df = pd.read_csv(RAW/filename, encoding='utf-8-sig', na_values=[' ', '  ', ''], skipinitialspace=True)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='raise')
    if df['Date'].duplicated().any():
        raise ValueError(f'Duplicate date in {filename}')
    return df

chem=load('Ballycanew_Turbidity_Sediments.csv')
flow=load('Ballycanew_discharge.csv')
trp=load('Ballycanew_data.csv')
merged=chem.merge(flow,on='Date',how='outer',validate='one_to_one').merge(trp,on='Date',how='outer',validate='one_to_one').sort_values('Date')
merged=merged.rename(columns={'Date':'date','Turbidity[NTU]':'turbidity_ntu','SuspendedSedimentLoad[g]':'suspended_sediment_load_g','Suspended Sediment[mg/l]':'suspended_sediment_mg_l','Qcumecs':'discharge_original_Qcumecs_units_unverified','TRP [mg/l]':'trp_mg_l'})
merged.insert(1,'site_id','BALLYCANEW_SOURCE_SERIES')
merged['source_type']='source_supplied_daily_observation_series'
merged.to_csv(OUT/'ballycanew_daily_merged.csv',index=False,date_format='%Y-%m-%d')

# Training candidates preserve daily chronology. A valid target must be exactly next calendar day.
frame=merged.sort_values('date').copy()
frame['next_date']=frame['date'].shift(-1)
frame['target_turbidity_ntu_t_plus_1']=frame['turbidity_ntu'].shift(-1)
frame['turbidity_ntu_t_minus_1']=frame['turbidity_ntu'].shift(1)
frame['discharge_t_minus_1_original']=frame['discharge_original_Qcumecs_units_unverified'].shift(1)
frame['prev_date']=frame['date'].shift(1)
valid=(frame['next_date']-frame['date']==pd.Timedelta(days=1)) & (frame['date']-frame['prev_date']==pd.Timedelta(days=1))
required=['turbidity_ntu','discharge_original_Qcumecs_units_unverified','turbidity_ntu_t_minus_1','discharge_t_minus_1_original','target_turbidity_ntu_t_plus_1']
frame=frame[valid & frame[required].notna().all(axis=1)].copy()
frame['split']=frame['date'].apply(lambda d: 'train' if d < pd.Timestamp('2015-01-01') else ('validation' if d < pd.Timestamp('2016-01-01') else 'test'))
frame[['date','site_id','split','turbidity_ntu','turbidity_ntu_t_minus_1','discharge_original_Qcumecs_units_unverified','discharge_t_minus_1_original','trp_mg_l','target_turbidity_ntu_t_plus_1']].to_csv(OUT/'ballycanew_next_day_supervised.csv',index=False,date_format='%Y-%m-%d')

report={'input_files': ['Ballycanew_Turbidity_Sediments.csv','Ballycanew_discharge.csv','Ballycanew_data.csv'],'all_calendar_rows':len(merged),'calendar_first':str(merged['date'].min().date()),'calendar_last':str(merged['date'].max().date()),'nonmissing_turbidity':int(merged['turbidity_ntu'].notna().sum()),'nonmissing_discharge':int(merged['discharge_original_Qcumecs_units_unverified'].notna().sum()),'nonmissing_trp':int(merged['trp_mg_l'].notna().sum()),'complete_turbidity_and_discharge':int(merged[['turbidity_ntu','discharge_original_Qcumecs_units_unverified']].notna().all(axis=1).sum()),'supervised_rows':len(frame),'supervised_splits':{k:int(v) for k,v in frame['split'].value_counts().items()},'notes':['The supplied files cover one catchment, not 5 independent catchments or 35 monitoring sites.','Qcumecs was preserved without unit conversion because the variable magnitude and units must be independently verified.','No missing values are filled.','Additional monthly-distribution files and lognormal_observed_TRP.csv are not treated as timestamped field measurements.','No river-network edges can be inferred from this single site series.','Valid supervised rows have one-day-separated adjacent samples and lagged features.']}
(OUT/'data_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
