"""Reproducible observed-data baselines, NOT federated GNN.
Fits standardized temporal Ridge regression and logistic alerts on Ballycanew.
Threshold (75th percentile) is derived from TRAIN targets only.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score,f1_score,roc_auc_score,log_loss

ROOT=Path(__file__).resolve().parents[1]
D=pd.read_csv(ROOT/'data/observed/processed/ballycanew_next_day_supervised.csv',parse_dates=['date'])
features=['turbidity_ntu','turbidity_ntu_t_minus_1','discharge_original_Qcumecs_units_unverified','discharge_t_minus_1_original','trp_mg_l']
assert D[features+['target_turbidity_ntu_t_plus_1']].notna().all().all()
train=D.split.eq('train').to_numpy(); val=D.split.isin(['val','validation']).to_numpy();test=D.split.eq('test').to_numpy()
assert (train.sum(),val.sum(),test.sum())==(1838,365,365)
assert D.date[train].max()<D.date[val].min()<D.date[test].min()
x=D[features].to_numpy(float);target=D.target_turbidity_ntu_t_plus_1.to_numpy(float)
threshold=float(np.quantile(target[train],.75));alerts=(target>threshold).astype(int)
models={'ridge':make_pipeline(StandardScaler(),Ridge(alpha=30)),
'logistic_alert':make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,random_state=42))}
models['ridge'].fit(x[train],target[train]);models['logistic_alert'].fit(x[train],alerts[train])
rows=[];predictions=pd.DataFrame({'date':D.date[test].dt.strftime('%Y-%m-%d').to_numpy(),'actual_ntu':target[test],'true_alert':alerts[test]})
for section,mask in [('validation',val),('test',test)]:
 yp=models['ridge'].predict(x[mask]);prob=models['logistic_alert'].predict_proba(x[mask])[:,1]
 rows.append({'dataset':'Ballycanew_observed','split':section,'model':'temporal_ridge','n':int(mask.sum()),
 'MAE_NTU':float(mean_absolute_error(target[mask],yp)),
 'RMSE_NTU':float(np.sqrt(mean_squared_error(target[mask],yp))),
 'alert_threshold_ntu':threshold})
 rows.append({'dataset':'Ballycanew_observed','split':section,'model':'logistic_alert','n':int(mask.sum()),
 'accuracy':float(accuracy_score(alerts[mask],prob>=.5)),
 'f1':float(f1_score(alerts[mask],prob>=.5,zero_division=0)),
 'ROC_AUC':float(roc_auc_score(alerts[mask],prob)) if len(set(alerts[mask]))==2 else None,
 'cross_entropy':float(log_loss(alerts[mask],prob,labels=[0,1])), 'alert_threshold_ntu':threshold})
 if section=='test': predictions['ridge_prediction_ntu']=yp;predictions['logistic_alert_probability']=prob
out=ROOT/'outputs/observed';out.mkdir(parents=True,exist_ok=True)
pd.DataFrame(rows).to_csv(out/'baseline_metrics.csv',index=False)
predictions.to_csv(out/'baseline_test_predictions.csv',index=False)
(out/'baseline_provenance.json').write_text(json.dumps({'data':'observed Ballycanew','input_features':features,'chronological_split_counts':[int(train.sum()),int(val.sum()),int(test.sum())],'threshold':threshold,'not_full_hydrograph_fl':True},indent=2))
print(pd.DataFrame(rows).to_string(index=False))
