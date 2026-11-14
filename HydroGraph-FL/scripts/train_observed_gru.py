from pathlib import Path
REPO_ROOT=Path(__file__).resolve().parents[1]
"""Train an honest single-node graph-compatible GRU classifier on Ballycanew observations.
A=I for the only available monitored series; graph layer does not represent river topology.
Reproducibility: seed=42, chronological split, training-only scaler, validation-selected checkpoint.
"""
import random, copy
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, confusion_matrix
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

seed=42; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1)
base=(REPO_ROOT/'outputs/observed');base.mkdir(exist_ok=True)
d=pd.read_csv(str(REPO_ROOT/'data/observed/processed/ballycanew_next_day_supervised.csv'),parse_dates=['date'])
train=d.split.eq('train').to_numpy(); val=d.split.eq('val').to_numpy(); test=d.split.eq('test').to_numpy()
print('splits',d.split.value_counts().to_dict()); print('label values',d.split.unique())
if not val.any(): val=d.split.eq('validation').to_numpy()
thr=float(d.loc[train,'target_turbidity_ntu_t_plus_1'].quantile(.75)); y=(d.target_turbidity_ntu_t_plus_1.to_numpy()>thr).astype('float32')
# 2 timesteps, 3 features each; TRP t-1 not measured in supplied supervised table, mark it 0 and add a separate mask flag.
cols=['turbidity_ntu_t_minus_1','discharge_t_minus_1_original','turbidity_ntu','discharge_original_Qcumecs_units_unverified','trp_mg_l']
raw=d[cols].to_numpy('float32')
sc=StandardScaler().fit(raw[train]); arr=sc.transform(raw).astype('float32')
x=np.stack([np.stack([arr[:,0],arr[:,1],np.zeros(len(d),dtype='float32'),np.zeros(len(d),dtype='float32')],-1),np.stack([arr[:,2],arr[:,3],arr[:,4],np.ones(len(d),dtype='float32')],-1)],1)
X=torch.tensor(x);Y=torch.tensor(y)
# Architecture GRU -> identity-graph propagation -> linear classifier.
class GraphCompatibleGRU(nn.Module):
  def __init__(self,width=32):
    super().__init__();self.gru=nn.GRU(input_size=4,hidden_size=width,batch_first=True);self.graph=nn.Linear(width,width);self.classifier=nn.Sequential(nn.ReLU(),nn.Dropout(.1),nn.Linear(width,1))
  def forward(self,x):
    h=self.gru(x)[0][:,-1,:]
    # Single-node adjacency A=[1]; incoming message is its own state, no river edges.
    z=torch.relu(self.graph(h))
    return self.classifier(z).squeeze(-1)

net=GraphCompatibleGRU(); optim=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.005);criterion=nn.BCEWithLogitsLoss()
tr_idx=np.where(train)[0]; history=[];best=float('inf');best_ep=0;best_st=None
for ep in range(1,121):
    net.train();ids=np.random.permutation(tr_idx)
    for batch in np.array_split(ids, max(1,int(np.ceil(len(ids)/64)))):
        optim.zero_grad(); l=criterion(net(X[batch]),Y[batch]);l.backward();optim.step()
    net.eval()
    with torch.no_grad():
        ptrain=torch.sigmoid(net(X[train])).numpy();pval=torch.sigmoid(net(X[val])).numpy()
    ltrain=log_loss(y[train],ptrain,labels=[0,1]);lval=log_loss(y[val],pval,labels=[0,1]);
    history.append({'epoch':ep,'train_accuracy':accuracy_score(y[train],ptrain>=.5),'validation_accuracy':accuracy_score(y[val],pval>=.5),'train_loss':ltrain,'validation_loss':lval})
    if lval<best-1e-6:best=lval;best_ep=ep;best_st=copy.deepcopy(net.state_dict())
net.load_state_dict(best_st);net.eval()
with torch.no_grad():p=torch.sigmoid(net(X[test])).numpy()
metrics={'model':'Single-monitor GRU with identity-graph layer (not river GNN)','threshold_ntu':thr,'best_epoch':best_ep,'test_accuracy':accuracy_score(y[test],p>=.5),'test_f1':f1_score(y[test],p>=.5,zero_division=0),'test_auc':roc_auc_score(y[test],p),'test_cross_entropy':log_loss(y[test],p,labels=[0,1]), 'confusion_matrix':confusion_matrix(y[test],p>=.5).tolist(), 'n_train':int(sum(train)),'n_validation':int(sum(val)),'n_test':int(sum(test))}
print(metrics)
hist=pd.DataFrame(history);hist.to_csv(base/'Ballycanew_GRU_SingleNode_Training_History.csv',index=False)
pd.DataFrame([{'date':z,'observed_alert':int(t),'predicted_probability':float(s),'predicted_alert':int(s>=.5)} for z,t,s in zip(d.loc[test,'date'],y[test],p)]).to_csv(base/'Ballycanew_GRU_SingleNode_Test_Predictions.csv',index=False)
import json
(base/'Ballycanew_GRU_SingleNode_Metrics.json').write_text(json.dumps(metrics,indent=2))
fig,axs=plt.subplots(1,2,figsize=(11,4.2));axs[0].plot(hist.epoch,hist.train_accuracy,label='Training');axs[0].plot(hist.epoch,hist.validation_accuracy,label='Validation');axs[0].set(xlabel='Epoch',ylabel='Accuracy',ylim=(0,1));axs[0].legend();axs[0].grid(alpha=.25)
axs[1].plot(hist.epoch,hist.train_loss,label='Training');axs[1].plot(hist.epoch,hist.validation_loss,label='Validation');axs[1].axvline(best_ep,linestyle=':',alpha=.7,label=f'Best validation epoch {best_ep}');axs[1].set(xlabel='Epoch',ylabel='Binary cross-entropy loss');axs[1].legend();axs[1].grid(alpha=.25)
fig.tight_layout();fig.savefig(base/'Ballycanew_GRU_SingleNode_Accuracy_Loss.png',dpi=300,bbox_inches='tight');plt.close(fig)
torch.save({'model_state':best_st,'threshold':thr,'architecture':'GRU(4,32) -> identity graph linear(32,32) -> ReLU -> dropout(0.1) -> classifier'},base/'Ballycanew_GRU_SingleNode_Model.pt')
