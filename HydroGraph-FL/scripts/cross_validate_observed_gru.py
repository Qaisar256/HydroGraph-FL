from pathlib import Path
REPO_ROOT=Path(__file__).resolve().parents[1]
"""Reproducible 10-fold forward-chaining CV for Ballycanew single-monitor GRU.
This is NOT a verified multi-node federated river GNN. The 2016 test period is untouched until final evaluation.
"""
import random, copy, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,log_loss, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED=20261114
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.set_num_threads(1)
root=(REPO_ROOT/'outputs/observed');root.mkdir(exist_ok=True)
d=pd.read_csv(str(REPO_ROOT/'data/observed/processed/ballycanew_next_day_supervised.csv'),parse_dates=['date']).sort_values('date').reset_index(drop=True)
cols=['turbidity_ntu_t_minus_1','discharge_t_minus_1_original','turbidity_ntu','discharge_original_Qcumecs_units_unverified','trp_mg_l']
raw=d[cols].to_numpy(dtype=np.float32); target=d.target_turbidity_ntu_t_plus_1.to_numpy(dtype=np.float32)
tr=(d.split=='train').to_numpy(); va=(d.split=='validation').to_numpy(); te=(d.split=='test').to_numpy(); development=np.flatnonzero(tr|va)
assert len(development)==2203 and te.sum()==365 and (d.date.iloc[development[-1]]<d.date[te].min())

class GRUClassifier(nn.Module):
 def __init__(self):
  super().__init__();self.gru=nn.GRU(4,24,batch_first=True);self.projection=nn.Sequential(nn.Linear(24,24),nn.ReLU(),nn.Dropout(.10),nn.Linear(24,1))
 def forward(self,x):
  return self.projection(self.gru(x)[0][:,-1,:]).squeeze(-1)

def build_x(train_indices):
 sc=StandardScaler().fit(raw[train_indices]);a=sc.transform(raw).astype('float32')
 x=np.stack([np.stack([a[:,0],a[:,1],np.zeros(len(d),np.float32),np.zeros(len(d),np.float32)],-1),np.stack([a[:,2],a[:,3],a[:,4],np.ones(len(d),np.float32)],-1)],1)
 return torch.from_numpy(x)

def score(y,p):
 return dict(accuracy=float(accuracy_score(y,p>=.5)),f1=float(f1_score(y,p>=.5,zero_division=0)),auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,loss=float(log_loss(y,np.clip(p,1e-7,1-1e-7),labels=[0,1])))

def fit(train_idx,val_idx,seed,max_epochs=200,patience=18):
 torch.manual_seed(seed);rng=np.random.default_rng(seed)
 X=build_x(train_idx);threshold=float(np.quantile(target[train_idx],.75)); Y=torch.from_numpy((target>threshold).astype('float32'))
 net=GRUClassifier(); opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.002);crit=nn.BCEWithLogitsLoss()
 best=1e9; best_epoch=0;bad=0;best_state=None;history=[]
 for ep in range(1,max_epochs+1):
  net.train();permutation=rng.permutation(train_idx)
  for batch in np.array_split(permutation,int(np.ceil(len(permutation)/64))):
   opt.zero_grad();loss=crit(net(X[batch]),Y[batch]);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step()
  net.eval()
  with torch.no_grad():
   pt=torch.sigmoid(net(X[train_idx])).numpy();pv=torch.sigmoid(net(X[val_idx])).numpy()
  yt=Y[train_idx].numpy();yv=Y[val_idx].numpy()
  st=score(yt,pt);sv=score(yv,pv)
  history.append({'epoch':ep,'train_accuracy':st['accuracy'],'validation_accuracy':sv['accuracy'],'train_loss':st['loss'],'validation_loss':sv['loss']})
  if sv['loss'] < best - 0.0002:best=sv['loss']; best_epoch=ep;bad=0;best_state=copy.deepcopy(net.state_dict())
  else:bad+=1
  if bad>=patience:break
 net.load_state_dict(best_state);net.eval()
 with torch.no_grad():pv=torch.sigmoid(net(X[val_idx])).numpy()
 result=score(Y[val_idx].numpy(),pv)
 return result,best_epoch,threshold,pd.DataFrame(history),net,X

splits=TimeSeriesSplit(n_splits=10,test_size=180,gap=1)
rows=[];histories=[]
for fold,(itr,iva) in enumerate(splits.split(development),1):
 train_idx=development[itr];val_idx=development[iva]
 met,ep,thr,hist,_,_=fit(train_idx,val_idx,SEED+fold)
 rows.append({'fold':fold,'train_start':str(d.date.iloc[train_idx[0]].date()),'train_end':str(d.date.iloc[train_idx[-1]].date()),'validation_start':str(d.date.iloc[val_idx[0]].date()),'validation_end':str(d.date.iloc[val_idx[-1]].date()),'n_train':len(train_idx),'n_validation':len(val_idx),'threshold_ntu':thr,'best_epoch':ep,**met})
 hist.insert(0,'fold',fold);histories.append(hist)
 print('fold',fold,'n',len(train_idx),len(val_idx),'bestep',ep,'acc',round(met['accuracy'],4),'loss',round(met['loss'],4),flush=True)
cv=pd.DataFrame(rows);cv.to_csv(root/'Ballycanew_10Fold_ForwardCV_Metrics.csv',index=False)
pd.concat(histories,ignore_index=True).to_csv(root/'Ballycanew_10Fold_ForwardCV_Histories.csv',index=False)
# Final model: choose early stopping using 2015 validation, then evaluate 2016 test once.
tridx=np.flatnonzero(tr);validx=np.flatnonzero(va);testidx=np.flatnonzero(te)
valmetric,epoch,thr,hist,net,X=fit(tridx,validx,SEED+999)
hist.to_csv(root/'Ballycanew_10Fold_Final_Training_History.csv',index=False)
with torch.no_grad(): prob=torch.sigmoid(net(X[testidx])).numpy()
truth=(target[testidx]>thr).astype(int)
testmetric=score(truth,prob)
conf=confusion_matrix(truth,prob>=.5).tolist()
majority=max(float(np.mean(truth)),1-float(np.mean(truth)))
result={'model':'GRU with single-monitor identity graph, NOT full HydroGraph-FL federated GRU-GNN','cv':'10-fold expanding-window, 1-day temporal gap; 180-day validation per fold, development data up to 2015','cv_accuracy_mean':float(cv.accuracy.mean()),'cv_accuracy_std':float(cv.accuracy.std(ddof=1)),'cv_f1_mean':float(cv.f1.mean()),'cv_loss_mean':float(cv.loss.mean()),'cv_auc_mean':float(cv.auc.mean()),'final_train_n':len(tridx),'final_validation_n':len(validx),'independent_test_n':len(testidx),'final_training_threshold_ntu':thr,'final_best_epoch':epoch,'final_val_metrics':valmetric,'test_metrics':testmetric,'test_confusion_matrix':conf,'test_majority_baseline_accuracy':majority,'cv_best_epochs':cv.best_epoch.tolist()}
(root/'Ballycanew_10Fold_ForwardCV_Summary.json').write_text(json.dumps(result,indent=2))
pd.DataFrame({'date':d.loc[te,'date'].dt.strftime('%Y-%m-%d').to_numpy(),'target_turbidity_ntu':target[testidx],'observed_alert':truth,'predicted_probability':prob,'predicted_alert':(prob>=.5).astype(int)}).to_csv(root/'Ballycanew_10Fold_Independent_Test_Predictions.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(11.8,4.3));axs[0].plot(cv.fold,cv.accuracy,marker='o',label='Validation accuracy');axs[0].plot(cv.fold,cv.f1,marker='s',label='Validation F1');axs[0].set(xlabel='Forward-chaining fold',ylabel='Score',ylim=(0,1));axs[0].legend(fontsize=9);axs[0].grid(alpha=.24)
axs[1].plot(cv.fold,cv.loss,marker='o',label='Validation binary cross-entropy');axs[1].set(xlabel='Forward-chaining fold',ylabel='Validation loss');axs[1].grid(alpha=.24)
fig.tight_layout();fig.savefig(root/'Ballycanew_10Fold_ForwardCV_Scores.png',dpi=350,bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(11.8,4.3));axs[0].plot(hist.epoch,hist.train_accuracy,label='Training');axs[0].plot(hist.epoch,hist.validation_accuracy,label='Validation');axs[0].set(xlabel='Epoch',ylabel='Accuracy',ylim=(0,1));axs[0].legend();axs[0].grid(alpha=.24)
axs[1].plot(hist.epoch,hist.train_loss,label='Training');axs[1].plot(hist.epoch,hist.validation_loss,label='Validation');axs[1].axvline(epoch,ls=':',color='gray',label='Selected checkpoint');axs[1].set(xlabel='Epoch',ylabel='Binary cross-entropy loss');axs[1].legend();axs[1].grid(alpha=.24)
fig.tight_layout();fig.savefig(root/'Ballycanew_10Fold_Final_Accuracy_Loss.png',dpi=350,bbox_inches='tight');plt.close(fig)
print('SUMMARY',json.dumps(result,indent=2),flush=True)
