"""Transparent, deliberately generated river-network sanity check.
NOT an evaluation of HydroGraph-FL neural-network implementation or WQP-USGS data.
Python 3; NumPy, pandas, scikit-learn, matplotlib, networkx.
Run: python scripts/hydrograph_generated_data_reproducible.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
DATA = ROOT / 'data'
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
rng=np.random.default_rng(20261109)
N_CLIENTS=5; N_SITES=7; T=540; START=80; TRAIN_END=375; VAL_END=455
Q=np.zeros((N_CLIENTS,N_SITES,T)); C=np.zeros_like(Q); P=np.zeros((N_CLIENTS,T)); TMP=np.zeros((N_CLIENTS,T));
for b in range(N_CLIENTS):
    climate=(b-2)*0.12
    for t in range(T):
        storm=(rng.random()<0.055)*(rng.uniform(2,7))
        P[b,t]=max(0,0.8+climate + 0.5*np.sin(2*np.pi*t/365)+storm+ rng.normal(0,.32))
        TMP[b,t]=18+4*np.cos(2*np.pi*(t+30)/365)+2*climate+rng.normal(0,.75)
    for s in range(N_SITES):
        Q[b,s,0]=7.0+1.2*b+0.5*s
        C[b,s,0]=10.0+1.5*b+0.35*s
    for t in range(1,T):
        for s in range(N_SITES):
            upq=Q[b,s-1,t-1] if s>0 else Q[b,s,t-1]
            upc=C[b,s-1,t-1] if s>0 else C[b,s,t-1]
            Q[b,s,t]=max(.15,1.1+(.72*Q[b,s,t-1]) + 1.5*P[b,t] + (0.06*upq if s>0 else 0) +rng.normal(0,0.9))
            base=2.3+0.45*b+0.12*s
            C[b,s,t]=max(.25,base+.62*C[b,s,t-1]+(.15*upc if s>0 else 0) + .15*Q[b,s,t-1] + .75*P[b,t-1]+.055*TMP[b,t-1]+rng.normal(0,1.4))

tup=['c_t','c_t_minus_1','q_t','q_t_minus_1','rain_t','temp_t']
tgraph=tup+['c_upstream_t','q_upstream_t']
rows=[]
for b in range(N_CLIENTS):
 for s in range(N_SITES):
  for t in range(START,T-1):
   up=s-1 if s>0 else s
   rows.append(dict(client=b+1,station=s+1,day=t,actual_ntu=C[b,s,t+1],
        c_t=C[b,s,t],c_t_minus_1=C[b,s,t-1],q_t=Q[b,s,t],q_t_minus_1=Q[b,s,t-1],
        rain_t=P[b,t], temp_t=TMP[b,t],c_upstream_t=C[b,up,t],q_upstream_t=Q[b,up,t],
        streamflow_m3s=Q[b,s,t],is_headwater=(s==0)))
df=pd.DataFrame(rows)
# Save every raw generated daily measurement and the station topology.
from datetime import datetime, timedelta
start_date = datetime(2024, 1, 1)
raw_rows = []
for b in range(N_CLIENTS):
    for s in range(N_SITES):
        for t in range(T):
            raw_rows.append(dict(catchment_id=f'C{b+1}', client=b+1,
                station_id=f'C{b+1}_S{s+1}', station=s+1, day=t,
                date=(start_date + timedelta(days=t)).strftime('%Y-%m-%d'),
                discharge_m3s=Q[b,s,t], turbidity_ntu=C[b,s,t],
                rainfall_mm=P[b,t], temperature_c=TMP[b,t],
                origin='computer_generated', seed=20261109))
pd.DataFrame(raw_rows).to_csv(DATA/'generated_station_daily_observations.csv', index=False)
pd.DataFrame([dict(catchment_id=f'C{b+1}', client=b+1,
    upstream_station_id=f'C{b+1}_S{s}', downstream_station_id=f'C{b+1}_S{s+1}',
    directed_edge=1) for b in range(N_CLIENTS) for s in range(1,N_SITES)]).to_csv(
    DATA/'directed_river_edges.csv', index=False)
df['station_id'] = 'C' + df.client.astype(str) + '_S' + df.station.astype(str)
df['target_day'] = df.day + 1
df['target_date'] = df.target_day.map(lambda d: (start_date + timedelta(days=int(d))).strftime('%Y-%m-%d'))
df['origin'] = 'computer_generated'
df['split'] = np.select([df.day < TRAIN_END, df.day < VAL_END], ['train','validation'], default='test')
df.to_csv(DATA/'forecast_supervised_rows.csv', index=False)
train=df[df.day<TRAIN_END].copy(); val=df[(df.day>=TRAIN_END)&(df.day<VAL_END)].copy(); test=df[df.day>=VAL_END].copy()
THRESHOLD=float(train.actual_ntu.quantile(.75))

models={}
def pred_all(name,model,cols,groups=False):
 if not groups:
  scaler=StandardScaler().fit(train[cols])
  xt=scaler.transform(train[cols]); xv=scaler.transform(val[cols]); xx=scaler.transform(test[cols])
  model.fit(xt,train.actual_ntu)
  models[name]=(model,scaler,cols)
  return model.predict(xx)

preds={}
preds['Persistence']=test.c_t.to_numpy().copy()
preds['Temporal Ridge']=pred_all('Temporal Ridge',Ridge(alpha=120),tup)
preds['Graph Ridge']=pred_all('Graph Ridge',Ridge(alpha=120),tgraph)
# Local-only client models on exactly each client's own observations
localpred=np.zeros(len(test)); localfits={}
for b in range(1,N_CLIENTS+1):
 ix=(test.client==b).to_numpy(); part=train[train.client==b]
 sc=StandardScaler().fit(part[tgraph]); mdl=Ridge(alpha=30).fit(sc.transform(part[tgraph]),part.actual_ntu)
 localpred[ix]=mdl.predict(sc.transform(test.loc[ix,tgraph])); localfits[b]=(mdl,sc)
preds['Local Graph Ridge']=localpred
# Federated averaging on a linear graph model (demonstrates client-wise communication, not GNN)
sc=StandardScaler().fit(train[tgraph])
xt=np.c_[np.ones(len(train)),sc.transform(train[tgraph])]; xv=np.c_[np.ones(len(val)),sc.transform(val[tgraph])]; xx=np.c_[np.ones(len(test)),sc.transform(test[tgraph])]
ytrain=train.actual_ntu.to_numpy(); yval=val.actual_ntu.to_numpy()
client_masks=[(train.client.to_numpy()==b) for b in range(1,N_CLIENTS+1)]
# standardize output by training statistics for smooth gradient descent
ym=float(ytrain.mean()); ys=float(ytrain.std()); yn=(ytrain-ym)/ys
weights=np.zeros(xt.shape[1]); best=None; best_mae=float('inf'); trace=[]
for rnd in range(1,141):
  ws=[]; ns=[]
  for mask in client_masks:
    xi=xt[mask]; yi=yn[mask]
    w=weights.copy()
    for _ in range(5):
      gradient=2*(xi.T@(xi@w-yi))/len(xi)
      gradient[1:]+=0.002*w[1:]
      w-=0.075*gradient
    ws.append(w); ns.append(len(xi))
  weights=np.average(np.vstack(ws),axis=0,weights=ns)
  vmae=mean_absolute_error(yval, (xv@weights)*ys+ym)
  trace.append([rnd,float(vmae)])
  if vmae<best_mae:best_mae=vmae;best=weights.copy();best_round=rnd
preds['FedAvg Graph Linear']=(xx@best)*ys+ym
pd.DataFrame(trace,columns=['round','validation_mae_ntu']).to_csv(OUT/'fedavg_validation_trace.csv',index=False)

true=test.actual_ntu.to_numpy(); flags=(true>THRESHOLD).astype(int)
metrics=[]
for name,p in preds.items():
 p=np.asarray(p); cl=(p>THRESHOLD).astype(int)
 try:auc=roc_auc_score(flags,p);prauc=average_precision_score(flags,p)
 except ValueError:auc=np.nan;prauc=np.nan
 metrics.append(dict(model=name,RMSE_NTU=float(np.sqrt(mean_squared_error(true,p))),MAE_NTU=float(mean_absolute_error(true,p)),R2=float(r2_score(true,p)),Accuracy=float(accuracy_score(flags,cl)),Precision=float(precision_score(flags,cl,zero_division=0)),Recall=float(recall_score(flags,cl,zero_division=0)),F1=float(f1_score(flags,cl,zero_division=0)),ROC_AUC=float(auc),PR_AUC=float(prauc),TP=int(confusion_matrix(flags,cl)[1,1]),FP=int(confusion_matrix(flags,cl)[0,1]),FN=int(confusion_matrix(flags,cl)[1,0]),TN=int(confusion_matrix(flags,cl)[0,0])))
met=pd.DataFrame(metrics); met.to_csv(OUT/'comparison_metrics.csv',index=False)
# Individual held-out predictions permit independently auditing every aggregate metric.
prediction_table = test[['client','station','station_id','day','target_day','target_date','actual_ntu','streamflow_m3s']].copy()
prediction_table['alert_true'] = flags
for model_name, p in preds.items():
    name = model_name.lower().replace(' ', '_').replace('-', '_')
    prediction_table[f'{name}_prediction_ntu'] = p
    prediction_table[f'{name}_alert_predicted'] = (np.asarray(p) > THRESHOLD).astype(int)
prediction_table.to_csv(OUT/'test_predictions.csv', index=False)
# A second, explicitly verified five-model classification comparison is available here.
met[['model','MAE_NTU','RMSE_NTU','Accuracy','F1','ROC_AUC']].to_csv(
    OUT/'table7_executed_surrogates_only.csv', index=False)


scar=[]
for pct in [20,40,60,80,100]:
 # restrict training to later contiguous part of training period (keeps the latest observed data)
 low=int(TRAIN_END-(TRAIN_END-START)*pct/100)
 part=train[train.day>=low]
 sx=StandardScaler().fit(part[tgraph]); lr=Ridge(alpha=120*pct/100).fit(sx.transform(part[tgraph]),part.actual_ntu)
 pred=lr.predict(sx.transform(test[tgraph])); cls=(pred>THRESHOLD).astype(int)
 scar.append(dict(training_fraction=pct,train_n=len(part),rmse_ntu=float(np.sqrt(mean_squared_error(true,pred))),mae_ntu=float(mean_absolute_error(true,pred)),accuracy=float(accuracy_score(flags,cls)),f1=float(f1_score(flags,cls,zero_division=0)),roc_auc=float(roc_auc_score(flags,pred))))
pd.DataFrame(scar).to_csv(OUT/'data_scarcity.csv',index=False)

clientrows=[]
for b in range(1,N_CLIENTS+1):
 ix=(test.client==b).to_numpy()
 for method in ['Local Graph Ridge','Graph Ridge','FedAvg Graph Linear']:
  p=preds[method][ix]; y=true[ix]
  clientrows.append(dict(client=b,model=method,n_test=len(y),mae_ntu=float(mean_absolute_error(y,p)),rmse_ntu=float(np.sqrt(mean_squared_error(y,p)))))
pd.DataFrame(clientrows).to_csv(OUT/'client_metrics.csv',index=False)

# High-flow regime performance, based on training quantile, excludes leakage
highq=train.streamflow_m3s.quantile(.90)
reg=[]
for condition,label in [(test.streamflow_m3s<=highq,'ordinary flow'),(test.streamflow_m3s>highq,'high flow (> training p90)')]:
 for method in ['Temporal Ridge','Graph Ridge','FedAvg Graph Linear']:
  ix=condition.to_numpy(); p=preds[method][ix]; y=true[ix]
  reg.append(dict(regime=label,model=method,n=int(ix.sum()),MAE_NTU=float(mean_absolute_error(y,p)),RMSE_NTU=float(np.sqrt(mean_squared_error(y,p)))))
pd.DataFrame(reg).to_csv(OUT/'flow_regime_metrics.csv',index=False)

# Generate NEW charts entirely from exact predictions, with explicit generated-data labelling
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.dpi':140})
fig,ax=plt.subplots(figsize=(9,4.0));
ind=np.arange(len(met)); ax.bar(ind,met.MAE_NTU);ax.set_xticks(ind,met.model,rotation=18,ha='right');ax.set_ylabel('MAE (generated-data NTU)');ax.set_title('Generated river network: one-day-ahead turbidity, chronological test');fig.tight_layout();fig.savefig(OUT/'generated-data_model_mae.png',dpi=230);plt.close(fig)
fig,ax=plt.subplots(figsize=(8.4,3.9)); z=pd.read_csv(OUT/'data_scarcity.csv');ax.plot(z.training_fraction,z.mae_ntu,'o-',label='MAE (NTU)');ax.plot(z.training_fraction,z.rmse_ntu,'s-',label='RMSE (NTU)');ax.set_xlabel('Contiguous training-history retained (%)');ax.set_ylabel('Generated error (NTU)');ax.set_title('Generated-data stress test: shortened training history');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'generated-data_scarcity.png',dpi=230);plt.close(fig)
fig,ax=plt.subplots(figsize=(8.5,4)); cm=pd.read_csv(OUT/'client_metrics.csv');
for k in ['Local Graph Ridge','Graph Ridge','FedAvg Graph Linear']:
 d=cm[cm.model==k];ax.plot(d.client,d.mae_ntu,'o-',label=k)
ax.set_xticks(range(1,6));ax.set_xlabel('Artificial catchment client');ax.set_ylabel('MAE (NTU)');ax.legend();ax.set_title('Same test rows for local and pooled/federated surrogates');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'generated-data_client_mae.png',dpi=230);plt.close(fig)
fig,ax=plt.subplots(figsize=(8.4,4));
for name in ['Temporal Ridge','Graph Ridge','FedAvg Graph Linear']:
 fp,tp,_=roc_curve(flags,preds[name]);ax.plot(fp,tp,label=f'{name} (AUC {roc_auc_score(flags,preds[name]):.3f})')
ax.plot([0,1],[0,1],'k:',lw=1);ax.set_xlabel('False positive rate');ax.set_ylabel('True positive rate');ax.legend(fontsize=9);ax.set_title('Generated-data turbidity alert, training-derived threshold');fig.tight_layout();fig.savefig(OUT/'generated-data_roc.png',dpi=230);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,3.5));ax.plot(np.array(trace)[:,0],np.array(trace)[:,1]);ax.axvline(best_round,linestyle=':',label=f'Best round {best_round}');ax.set_xlabel('FedAvg communication round');ax.set_ylabel('Validation MAE (generated-data NTU)');ax.set_title('Generated-data FedAvg linear-surrogate training trace');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(OUT/'generated-data_fedavg_convergence.png',dpi=230);plt.close(fig)
fig,ax=plt.subplots(figsize=(9.4,3));
g=nx.DiGraph();
for b in range(1,6):
 for s in range(1,8):
  node=f'{b}-{s}';g.add_node(node,pos=(s,b))
  if s>1:g.add_edge(f'{b}-{s-1}',node)
pos=nx.get_node_attributes(g,'pos');nx.draw_networkx(g,pos,ax=ax,with_labels=False,node_size=90,arrows=True,arrowsize=12,width=1.3)
ax.set_xlim(0,8);ax.set_ylim(.35,5.65);ax.set_title('Generated-data directed river chains: 5 isolated clients, 7 stations each');ax.axis('off');fig.tight_layout();fig.savefig(OUT/'generated-data_network.png',dpi=230);plt.close(fig)

metadata=dict(description='Generated river-network sanity-check only. Not an execution of full HydroGraph-FL and not WQP-USGS field validation.',seed=20261109,clients=5,stations_per_client=7,days=T,record_generation_start=START,train_before_day=TRAIN_END,validation_before_day=VAL_END,train_records=int(len(train)),validation_records=int(len(val)),test_records=int(len(test)),full_daily_observations=int(N_CLIENTS*N_SITES*T),forecast_supervised_rows=int(len(df)),train_binary_threshold_NTU=THRESHOLD,test_positives=int(flags.sum()),test_negatives=int(len(flags)-flags.sum()),fedavg_rounds=140,selected_fedavg_round=int(best_round),features_temporal=tup,features_graph=tgraph,restriction='All predictors from day t or earlier; target day t+1. Graph edges only within clients; data-generation parameters fixed, all outputs generated.')
(OUT/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
(DATA/'README_DATA.txt').write_text('All rows in these CSV files are COMPUTER-GENERATED, not WQP/USGS measurements.\nDaily observations: 5 catchments x 7 stations x 540 days = 18,900 records.\nForecast features use day t and predict turbidity at t+1.\nChronological split by feature day: train 80-374; validation 375-454; test 455-538.\nThe 90th-percentile discharge boundary and 75th-percentile alert threshold are fitted on training data only.\n')
print('METADATA',metadata)
print(met.round(4).to_string(index=False));print('SCARCITY');print(pd.DataFrame(scar).round(3).to_string(index=False))
pd.DataFrame({'day':np.arange(T),'discharge_m3s':Q[0,3,:],'turbidity_ntu':C[0,3,:]}).to_csv(OUT/'example_hydrograph.csv',index=False)
fig,ax=plt.subplots(figsize=(8.5,4));
for k in ['Local Graph Ridge','Graph Ridge','FedAvg Graph Linear']:
 d=cm[cm.model==k];ax.plot(d.client,d.rmse_ntu,'o-',label=k)
ax.set_xticks(range(1,6));ax.set_xlabel('Artificial catchment client');ax.set_ylabel('RMSE (NTU)');ax.legend();ax.set_title('Generated local and pooled/federated errors, same test rows');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'generated-data_client_rmse.png',dpi=230);plt.close(fig)
# Additional visual diagnostics
fig,ax=plt.subplots(figsize=(6.1,4.3));matrix=confusion_matrix(flags,(preds['Graph Ridge']>THRESHOLD).astype(int));img=ax.imshow(matrix,cmap='Blues');ax.set_xticks([0,1],['Below threshold','High turbidity']);ax.set_yticks([0,1],['Below threshold','High turbidity']);ax.set_xlabel('Predicted');ax.set_ylabel('Observed');ax.set_title('Generated-data Graph Ridge alert counts, n=2,940');
for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>matrix.max()*.5 else 'black',fontsize=13)
fig.tight_layout();fig.savefig(OUT/'generated-data_confusion.png',dpi=230);plt.close(fig)
fig,ax=plt.subplots(figsize=(8.6,3.9));ax.bar(np.arange(len(met)),100*met.Accuracy);ax.set_xticks(np.arange(len(met)),met.model,rotation=17,ha='right');ax.set_ylabel('Alert accuracy (%)');ax.set_ylim(88,96);ax.set_title('Generated-data next-day high-turbidity alert accuracy');fig.tight_layout();fig.savefig(OUT/'generated-data_accuracy.png',dpi=230);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,3.9));dd=pd.read_csv(OUT/'flow_regime_metrics.csv');
for method in ['Temporal Ridge','Graph Ridge','FedAvg Graph Linear']:
    sub=dd[dd.model==method];ax.plot([0,1],sub.MAE_NTU,'o-',label=method)
ax.set_xticks([0,1],['Ordinary flow','High flow (> training p90)']);ax.set_ylabel('MAE (generated-data NTU)');ax.set_title('Flow-regime error comparison, generated-data test');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(OUT/'generated-data_flow_regime.png',dpi=230);plt.close(fig)
