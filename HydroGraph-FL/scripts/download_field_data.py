#!/usr/bin/env python3
"""Download USGS daily discharge and WQP discrete turbidity; preserve provenance.

Daily values use USGS modern OGC API, not the retiring NWIS WaterServices.
No interpolation, fake observations, or unverified edges.
"""
from __future__ import annotations
import argparse
import io
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DAILY_URL = 'https://api.waterdata.usgs.gov/ogcapi/v1/collections/daily/items'
WQP_URL = 'https://www.waterqualitydata.us/wqx3/Result/search'
WQP_FALLBACK = 'https://www.waterqualitydata.us/data/Result/search'
ALLOWED_HOSTS = {'api.waterdata.usgs.gov', 'www.waterqualitydata.us', 'waterqualitydata.us'}
LOG = logging.getLogger(__name__)


def fetch(url, params=None, *, timeout=120, retries=4):
    if urlparse(url).hostname not in ALLOWED_HOSTS:
        raise ValueError('Unexpected host in data download URL: '+url)
    headers = {'User-Agent':'HydroGraph-FL/1.1 (academic open-water-data client)', 'Accept':'application/json,text/csv'}
    token=os.getenv('USGS_API_KEY','').strip()
    if token and urlparse(url).hostname == 'api.waterdata.usgs.gov': headers['X-Api-Key']=token
    for attempt in range(retries):
        try:
            res=requests.get(url,params=params,headers=headers,timeout=timeout)
            res.raise_for_status()
            if not res.content: raise ValueError('Empty response from '+res.url)
            if res.content[:300].lower().lstrip().startswith((b'<html',b'<!doctype')):
                raise ValueError('Unexpected HTML response at '+res.url)
            return res
        except (requests.RequestException,ValueError) as exc:
            if attempt == retries-1: raise RuntimeError(f'Failed to fetch {url}: {exc}') from exc
            time.sleep(2**attempt)


def parse_usgs_daily(payload, site):
    if isinstance(payload,bytes):payload=json.loads(payload)
    output=[]
    for feature in payload.get('features',[]):
        p=feature.get('properties',{})
        if str(p.get('monitoring_location_id','')) not in (f'USGS-{site}',site): continue
        if str(p.get('parameter_code'))!='00060' or str(p.get('statistic_id'))!='00003': continue
        unit=str(p.get('unit_of_measure','')).lower().replace(' ','')
        if unit and unit not in ('ft^3/s','ft3/s','cfs','ft³/s'):
            raise ValueError(f'Unexpected streamflow unit at {site}: {unit}')
        value=pd.to_numeric(p.get('value'),errors='coerce')
        if pd.isna(value) or value<0:continue
        output.append({'site_id':site,'date':str(p.get('time',''))[:10],
                       'discharge_cfs':float(value), 'discharge_m3s':float(value)*0.028316846592,
                       'approval_status':'|'.join(p.get('approvals_status') or []),
                       'source':'USGS OGC daily 00060/00003'})
    cols=['site_id','date','discharge_cfs','discharge_m3s','approval_status','source']
    if not output:return pd.DataFrame(columns=cols)
    result=pd.DataFrame(output)
    if result.duplicated(['site_id','date']).any():
        raise ValueError('Duplicate USGS daily data: inspect overlapping time series rather than choosing silently')
    return result[cols].sort_values(['site_id','date']).reset_index(drop=True)


def download_usgs_daily(site, start, end, rawdir):
    """Follow pagination via official links; protect against accidental unbounded download."""
    params={'f':'json','monitoring_location_id':f'USGS-{site}', 'parameter_code':'00060',
            'statistic_id':'00003','datetime':f'{start}/{end}','limit':10000}
    url=DAILY_URL;parts=[];seen=set(); pages=0;links=[]
    while url:
        res=fetch(url,params if pages==0 else None)
        data=res.json()
        pages+=1
        if pages>100 or res.url in seen:raise RuntimeError('USGS pagination unexpectedly repeated or exceeded 100 pages')
        seen.add(res.url);links.append(res.url)
        if pages==1: (rawdir/f'usgs_daily_{site}_first_page.json').write_bytes(res.content)
        parts.append(parse_usgs_daily(data,site))
        nxt=next((x.get('href') for x in data.get('links',[]) if x.get('rel')=='next'),None)
        url=nxt;params=None
    out=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
    if len(out) and out.duplicated(['site_id','date']).any():
        raise ValueError('Overlapping pages or time series led to duplicate daily observations')
    return out,links


def optional_column(df,candidates):
    names={''.join(c.lower() for c in str(col) if c.isalnum()):col for col in df.columns}
    for candidate in candidates:
        key=''.join(c.lower() for c in candidate if c.isalnum())
        if key in names:return names[key]
    return None


def parse_wqp_csv(payload, site):
    df=pd.read_csv(io.BytesIO(payload),dtype=str,low_memory=False)
    cols=['site_id','date','turbidity_value','turbidity_unit','activity_id','method_id','source']
    if df.empty:return pd.DataFrame(columns=cols)
    mapping={
        'site_id':['MonitoringLocationIdentifier','MonitoringLocationID','MonitoringLocationId'],
        'date':['ActivityStartDate','ActivityStartDateTime'],
        'turbidity_value':['ResultMeasureValue','ResultResultMeasureValue','ResultValue'],
        'turbidity_unit':['ResultMeasure/MeasureUnitCode','ResultMeasureMeasureUnitCode','ResultMeasureUnitCode','ResultUnit'],
        'activity_id':['ActivityIdentifier','ActivityID'],
        'method_id':['ResultAnalyticalMethod/MethodIdentifier','ResultAnalyticalMethodMethodIdentifier'],
        'characteristic':['CharacteristicName','ResultCharacteristicName']}
    col={k:optional_column(df,v) for k,v in mapping.items()}
    if not col['date'] or not col['turbidity_value'] or not col['turbidity_unit']:
        raise ValueError('WQP result profile is missing required date/value/unit columns. Columns: '+repr(list(df.columns)[:30]))
    out=pd.DataFrame(index=df.index)
    out['site_id']=df[col['site_id']].astype(str).str.replace(r'^USGS-','',regex=True) if col['site_id'] else site
    out['date']=pd.to_datetime(df[col['date']],errors='coerce').dt.strftime('%Y-%m-%d')
    out['turbidity_value']=pd.to_numeric(df[col['turbidity_value']],errors='coerce')
    out['turbidity_unit']=df[col['turbidity_unit']].astype(str).str.upper().str.strip()
    out['activity_id']=df[col['activity_id']].fillna('') if col['activity_id'] else ''
    out['method_id']=df[col['method_id']].fillna('') if col['method_id'] else ''
    if col['characteristic']:
        out=out[df[col['characteristic']].fillna('').str.contains('turbidity',case=False)]
    out=out[out.site_id.eq(site)]
    out=out[out.turbidity_unit.isin(['NTU','FNU','NTRU'])]
    out=out.dropna(subset=['date','turbidity_value'])
    out=out[out.turbidity_value>=0]
    out['source']='Water Quality Portal discrete sample'
    return out[cols].sort_values(['site_id','date','activity_id']).reset_index(drop=True)


def download_wqp(site,start,end,rawdir,mode='auto'):
    endpoints=[WQP_URL,WQP_FALLBACK] if mode=='auto' else ([WQP_URL] if mode=='wqx3' else [WQP_FALLBACK])
    errors=[]
    params={'siteid':f'USGS-{site}','characteristicName':'Turbidity',
            'startDateLo':pd.Timestamp(start).strftime('%m-%d-%Y'),
            'startDateHi':pd.Timestamp(end).strftime('%m-%d-%Y'),
            'mimeType':'csv','zip':'no','dataProfile':'fullPhysChem'}
    for endpoint in endpoints:
        try:
            r=fetch(endpoint,params)
            parsed=parse_wqp_csv(r.content,site)
            (rawdir/f'wqp_discrete_turbidity_{site}.csv').write_bytes(r.content)
            return parsed,r.url
        except (requests.RequestException,RuntimeError,ValueError,pd.errors.ParserError) as exc:
            errors.append(f'{endpoint}: {exc}')
    raise RuntimeError('; '.join(errors))


def join_exact_dates(flow,turb):
    if flow.empty or turb.empty:return pd.DataFrame(columns=['site_id','date','turbidity_value','turbidity_unit','discharge_m3s'])
    return turb.merge(flow,on=['site_id','date'],how='inner',validate='many_to_one').sort_values(['site_id','date'])


def daily_calendar(flow,turb,start,end,site):
    calendar=pd.DataFrame({'date':pd.date_range(start,end,freq='D').strftime('%Y-%m-%d')})
    calendar['site_id']=site
    calendar=calendar.merge(flow,on=['site_id','date'],how='left',validate='one_to_one')
    for unit,group in turb.groupby('turbidity_unit'):
        med=group.groupby(['site_id','date'],as_index=False)['turbidity_value'].median()
        med=med.rename(columns={'turbidity_value':f'turbidity_{unit.lower()}_observed_sample_median'})
        calendar=calendar.merge(med,on=['site_id','date'],how='left',validate='one_to_one')
    return calendar


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--start',default='2019-01-01');p.add_argument('--end',default='2025-12-31')
    p.add_argument('--sites',nargs='*');p.add_argument('--wqp-mode',choices=['auto','wqx3','legacy'],default='auto')
    p.add_argument('--continue-on-error',action='store_true')
    a=p.parse_args(argv)
    if pd.Timestamp(a.start)>pd.Timestamp(a.end):p.error('start must precede end')
    raw=ROOT/'data'/'field_raw';out=ROOT/'data'/'field_processed'
    raw.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    stations=a.sites or pd.read_csv(ROOT/'config'/'stations.csv',dtype=str).site_id.tolist()
    flows=[];turbs=[];matches=[];calendars=[];meta=[];errors=[]
    for site in stations:
        LOG.info('Retrieving %s',site)
        try:
            f,urls=download_usgs_daily(site,a.start,a.end,raw)
            t,wurl=download_wqp(site,a.start,a.end,raw,a.wqp_mode)
            flows.append(f);turbs.append(t);matches.append(join_exact_dates(f,t))
            calendars.append(daily_calendar(f,t,a.start,a.end,site))
            meta.append({'site_id':site,'usgs_query_urls':urls,'wqp_query_url':wurl,
                         'discharge_days':len(f),'discrete_samples':len(t)})
        except Exception as exc:
            LOG.error('%s: %s',site,exc);errors.append({'site_id':site,'reason':str(exc)})
            if not a.continue_on_error:raise
    if not meta:raise RuntimeError('No station downloaded successfully. Files were not substituted with generated data.')
    counts={}
    for frames,name in [(flows,'daily_discharge_all_sites.csv'),(turbs,'discrete_turbidity_all_sites.csv'),
                        (matches,'matched_turbidity_discharge.csv'),(calendars,'daily_calendar_with_missingness.csv')]:
        df=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
        df.to_csv(out/name,index=False);counts[name]=len(df)
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'start':a.start,'end':a.end,
              'sites':meta,'errors':errors,'counts':counts,'data_origin':'field_observations',
              'flow_endpoint':DAILY_URL,'wqp_endpoint':WQP_URL,
              'topology_verified':False,'turbidity_imputed':False,
              'note':'WQP discrete turbidity is not a complete daily series; no missing value inferred.'}
    (out/'download_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')
    main()
