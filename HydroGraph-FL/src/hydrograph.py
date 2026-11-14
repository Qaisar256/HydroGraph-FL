"""Full research implementation: GRU + directed edge-aware GNN + conditional physics + FL + QDE.
Data are computer-generated. Neural-model metrics must be obtained from actual runs.
"""
from __future__ import annotations
import copy, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

FEATURES=['turbidity_ntu','discharge_m3s','rainfall_mm','temperature_c']

def seed_everything(seed=20261109):
    random.seed(seed); np.random.seed(seed%2**32); torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))

def load_arrays(data_dir,window=14):
    """One sample represents all seven stations of one catchment at an anchor day.
    Targets are day t+1, with split inherited from original auditable supervised CSV.
    Normalization fitted ONLY using training dates and training targets.
    """
    data_dir=Path(data_dir)
    raw=pd.read_csv(data_dir/'generated_station_daily_observations.csv')
    labels=pd.read_csv(data_dir/'forecast_supervised_rows.csv')
    edge=pd.read_csv(data_dir/'directed_river_edges.csv')
    physics_file=data_dir/'observed_physics_inputs.csv'
    physics_table={}
    physics_columns=['q_out_m3s','upstream_sediment_flux_kg_s','source_sink_kg_s','storage_change_kg_s','calibration_slope_mg_l_per_ntu','calibration_intercept_mg_l','physical_scale_kg_s','valid']
    if physics_file.exists():
        phys_df=pd.read_csv(physics_file)
        if len(phys_df):
            for col in ['client','day','station']+physics_columns:
                if col not in phys_df.columns:raise ValueError(f'Missing observed physics CSV column: {col}')
            for _,pr in phys_df.iterrows():
                physics_table[(int(pr.client),int(pr.day),int(pr.station))]=pr
    raw=raw.sort_values(['client','station','day'])
    trains=labels[labels.split=='train']
    train_days=set(trains.day.unique().tolist())
    stats_raw=raw[raw.day.isin(train_days)]
    mu=stats_raw[FEATURES].mean().to_numpy(dtype=np.float32)
    sd=stats_raw[FEATURES].std().clip(lower=1.e-6).to_numpy(dtype=np.float32)
    ym=float(trains.actual_ntu.mean()); ys=float(max(trains.actual_ntu.std(),1.e-6))
    all_samples=[]; adjs={}; station_order={}
    for client,group in raw.groupby('client',sort=True):
        group=group.sort_values(['station','day'])
        stations=sorted(group.station.unique()); station_order[int(client)]=stations
        arr=np.stack([group[group.station==s].sort_values('day')[FEATURES].to_numpy(dtype=np.float32) for s in stations])
        days=group[group.station==stations[0]].sort_values('day').day.to_numpy()
        if any(not np.array_equal(group[group.station==s].sort_values('day').day.to_numpy(),days) for s in stations):
            raise ValueError('Station-day grid is incomplete')
        look={int(day):i for i,day in enumerate(days)}
        adj=np.zeros((len(stations),len(stations)),np.float32)
        ed=edge[edge.client==client]
        for _,e in ed.iterrows():
            src=stations.index(int(str(e.upstream_station_id).split('_S')[-1])); dst=stations.index(int(str(e.downstream_station_id).split('_S')[-1]))
            adj[dst,src]=1.0 # row=recipient/downstream, col=sender/upstream
        adjs[int(client)]=adj
        l=labels[labels.client==client]
        for day,one_day in l.groupby('day',sort=True):
            day=int(day)
            if day not in look or day-window+1 not in look or day+1 not in look:continue
            idx=look[day]
            if idx-window+1<0 or idx+1>=len(days):continue
            if not np.all(np.diff(days[idx-window+1:idx+1])==1):continue
            one_day=one_day.sort_values('station')
            if len(one_day)!=len(stations):raise ValueError(f'Incomplete client-day {client}/{day}')
            y=one_day.actual_ntu.to_numpy(dtype=np.float32)
            split=one_day.split.iloc[0]
            if any(one_day.split != split):raise ValueError('Split disagrees across stations')
            physical=np.zeros((len(stations),len(physics_columns)),dtype=np.float32)
            for j,station in enumerate(stations):
                record=physics_table.get((int(client),day,int(station)))
                if record is not None:
                    physical[j,:]=np.asarray([record[key] for key in physics_columns],dtype=np.float32)
            all_samples.append({'client':int(client),'day':day,'split':str(split),'physics':physical,
                'x':((arr[:,idx-window+1:idx+1,:]-mu)/sd).astype('float32'),
                'y':((y-ym)/ys).astype('float32'),'y_raw':y,
                'flow':arr[:,idx,1].copy(),'adj':adj,
                'station_ids':[f'C{client}_S{s}' for s in stations]})
    return all_samples,{'feature_mean':mu.tolist(),'feature_sd':sd.tolist(),'target_mean':ym,'target_sd':ys,'window':window,'station_order':station_order}

class RiverDataset(Dataset):
    def __init__(self,samples):self.rows=samples
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        return dict(x=torch.from_numpy(r['x']),y=torch.from_numpy(r['y']),adj=torch.from_numpy(r['adj']),physics=torch.from_numpy(r['physics']),client=torch.tensor(r['client']),day=torch.tensor(r['day']))

def loader(rows,batch=16,shuffle=False):
    return DataLoader(RiverDataset(rows),batch_size=batch,shuffle=shuffle,num_workers=0)

class DirectedGraphLayer(nn.Module):
    """Edge-aware messages from upstream to downstream; heads are source-only."""
    def __init__(self,dim,dropout):
        super().__init__()
        self.message=nn.Sequential(nn.Linear(2*dim+1,dim),nn.GELU(),nn.Linear(dim,dim))
        self.gate=nn.Sequential(nn.Linear(2*dim+1,dim),nn.Sigmoid())
        self.norm=nn.LayerNorm(dim)
        self.dropout=nn.Dropout(dropout)
    def forward(self,h,adj,flow):
        b,n,d=h.shape
        dst=h.unsqueeze(2).expand(b,n,n,d)
        src=h.unsqueeze(1).expand(b,n,n,d)
        flow_pair=torch.log1p(torch.abs(flow)).unsqueeze(1).expand(b,n,n)
        pair=torch.cat([dst,src,flow_pair.unsqueeze(-1)],-1)
        mask=adj.unsqueeze(-1)
        msgs=self.message(pair)*self.gate(pair)*mask
        denom=mask.sum(dim=2).clamp(min=1)
        aggregated=msgs.sum(dim=2)/denom
        return self.norm(h+self.dropout(aggregated))

class HydroGraphFL(nn.Module):
    def __init__(self,input_dim=4,hidden=64,layers=2,gnn_layers=2,dropout=.15):
        super().__init__()
        self.gru=nn.GRU(input_dim,hidden,num_layers=layers,batch_first=True,dropout=dropout if layers>1 else 0)
        self.graph=nn.ModuleList(DirectedGraphLayer(hidden,dropout) for _ in range(gnn_layers))
        self.readout=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,1))
    def forward(self,x,adj):
        b,n,t,f=x.shape
        h,_=self.gru(x.reshape(b*n,t,f))
        h=h[:,-1,:].reshape(b,n,-1)
        # standardized log-flow input is used only as a learned edge attribute, not a physical flux
        flow=x[:,:,-1,1]
        for layer in self.graph:h=layer(h,adj,flow)
        return self.readout(h).squeeze(-1)

def physics_flux_loss(pred_raw,physics):
    """Sediment mass-balance loss, kg/s: Qout * C(pred) - upstream_flux
    - net_external_sources + storage_change. Requires externally calibrated
    SSC(mg/L) = a * turbidity(NTU) + b, with site-specific verified a,b.
    Conversion m3/s * mg/L to kg/s is 0.001. Missing data are masked.
    No such observations or calibration exist in included synthetic CSVs.
    """
    if physics is None:return pred_raw.sum()*0,0
    mask=physics[...,7].bool()
    num=int(mask.sum().item())
    if not num:return pred_raw.sum()*0,0
    q,up,source,storage,slope,intercept,scale=[physics[...,i] for i in range(7)]
    ssc=torch.clamp(pred_raw*slope+intercept,min=0)
    residual=q*ssc*.001-up-source+storage
    loss=((residual/scale.clamp(min=1.e-6))[mask]**2).mean()
    return loss,num

def get_physics(batch,device='cpu'):
    return batch['physics'].to(device) if 'physics' in batch else None

def objective(model,batch,scaler,physics_weight=0,device='cpu'):
    x=batch['x'].to(device); y=batch['y'].to(device); adj=batch['adj'].to(device)
    pred=model(x,adj); fit=((pred-y)**2).mean()
    p_loss,n=physics_flux_loss(pred*scaler['target_sd']+scaler['target_mean'],get_physics(batch,device))
    return fit+physics_weight*p_loss,fit.detach(),n

@torch.no_grad()
def evaluate(model,rows,scaler,batch_size=32,device='cpu',threshold=None):
    model.eval(); yt=[];yp=[];metadata=[]
    for batch in loader(rows,batch_size):
        pr=model(batch['x'].to(device),batch['adj'].to(device)).cpu().numpy()*scaler['target_sd']+scaler['target_mean']
        actual=batch['y'].numpy()*scaler['target_sd']+scaler['target_mean']
        yt.extend(actual.reshape(-1).tolist());yp.extend(pr.reshape(-1).tolist())
        for c,day in zip(batch['client'].numpy(),batch['day'].numpy()):
            metadata.extend([(int(c),int(day),i+1) for i in range(pr.shape[1])])
    yt=np.asarray(yt);yp=np.asarray(yp)
    from sklearn.metrics import mean_absolute_error,mean_squared_error,accuracy_score,f1_score,roc_auc_score
    met={'n':len(yt),'mae_ntu':float(mean_absolute_error(yt,yp)),'rmse_ntu':float(np.sqrt(mean_squared_error(yt,yp)))}
    if threshold is not None:
        a=(yt>=threshold).astype(int);p=(yp>=threshold).astype(int)
        met['alert_accuracy_pct']=float(accuracy_score(a,p)*100)
        met['alert_f1_pct']=float(f1_score(a,p,zero_division=0)*100)
        met['alert_auc']=float(roc_auc_score(a,yp)) if len(np.unique(a))==2 else None
    df=pd.DataFrame(metadata,columns=['client','anchor_day','station'])
    df['actual_ntu']=yt;df['predicted_ntu']=yp
    if threshold is not None:df['actual_alert']=(yt>=threshold).astype(int);df['predicted_alert']=(yp>=threshold).astype(int)
    return met,df

def train_central(train,val,scaler,config,out_dir,device='cpu',label='central'):
    seed_everything(int(config.get('seed',20261109)))
    model=HydroGraphFL(hidden=int(config['hidden']),layers=int(config.get('gru_layers',2)),
        gnn_layers=int(config.get('gnn_layers',2)),dropout=float(config['dropout'])).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=float(config['lr']),weight_decay=float(config.get('weight_decay',1e-4)))
    steps=int(config.get('epochs',30));patience=int(config.get('patience',7))
    best=np.inf;best_state=None;wait=0;history=[]
    for epoch in range(steps):
        model.train();losses=[];valid_physics=0
        for batch in loader(train,int(config['batch']),shuffle=True):
            opt.zero_grad();loss,_,n=objective(model,batch,scaler,float(config.get('physics_weight',0)),device)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step();losses.append(loss.item());valid_physics+=n
        val_metrics,_=evaluate(model,val,scaler,device=device)
        history.append(dict(epoch=epoch+1,train_scaled_mse=float(np.mean(losses)),validation_mae_ntu=val_metrics['mae_ntu'],valid_physics_constraints=valid_physics))
        if val_metrics['mae_ntu']<best-1e-7:best=val_metrics['mae_ntu'];best_state=copy.deepcopy(model.state_dict());wait=0
        else:
            wait+=1
            if wait>=patience:break
    model.load_state_dict(best_state)
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':model.state_dict(),'config':config,'scaler':scaler},out_dir/f'{label}.pt')
    pd.DataFrame(history).to_csv(out_dir/f'{label}_training_history.csv',index=False)
    return model,best

def fed_train(train,val,scaler,config,out_dir,device='cpu',algorithm='fedavg'):
    seed_everything(int(config.get('seed',20261109)))
    model=HydroGraphFL(hidden=int(config['hidden']),layers=int(config.get('gru_layers',2)),
        gnn_layers=int(config.get('gnn_layers',2)),dropout=float(config['dropout'])).to(device)
    clients=sorted({r['client'] for r in train});rounds=int(config.get('rounds',10));local_epochs=int(config.get('local_epochs',1))
    best=np.inf;best_state=None;history=[]
    for rnd in range(rounds):
        global_state=copy.deepcopy(model.state_dict());local_states=[];weights=[]
        for c in clients:
            rows=[r for r in train if r['client']==c]
            local=copy.deepcopy(model)
            opt=torch.optim.AdamW(local.parameters(),lr=float(config['lr']),weight_decay=float(config.get('weight_decay',1e-4)))
            for _ in range(local_epochs):
                for batch in loader(rows,int(config['batch']),shuffle=True):
                    opt.zero_grad();loss,_,_=objective(local,batch,scaler,float(config.get('physics_weight',0)),device)
                    if algorithm=='fedprox':
                        mu=float(config.get('prox_mu',0.01))
                        reg=sum(((p-global_state[n].to(device))**2).sum() for n,p in local.named_parameters())
                        loss=loss+mu*.5*reg
                    loss.backward();nn.utils.clip_grad_norm_(local.parameters(),2);opt.step()
            local_states.append({k:v.detach().cpu() for k,v in local.state_dict().items()});weights.append(len(rows))
        total=sum(weights)
        merged={k:sum(local_states[j][k]*(weights[j]/total) for j in range(len(clients))) for k in global_state}
        model.load_state_dict(merged)
        met,_=evaluate(model,val,scaler,device=device)
        history.append({'round':rnd+1,'validation_mae_ntu':met['mae_ntu'],'clients':len(clients),'communication_parameters':sum(p.numel() for p in model.parameters())*len(clients)*2})
        if met['mae_ntu']<best:best=met['mae_ntu'];best_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':model.state_dict(),'config':config,'scaler':scaler,'algorithm':algorithm},out_dir/f'{algorithm}.pt')
    pd.DataFrame(history).to_csv(out_dir/f'{algorithm}_training_history.csv',index=False)
    return model,best

def qde_optimize(train,val,scaler,base_cfg,out_dir,device='cpu',population=4,generations=2,trial_epochs=2):
    """Quantum-inspired bounded differential evolution in angle space.
    A real-valued angle theta encodes p=sin(theta)^2; differential mutation,
    crossover and rotation towards the best angle update candidate amplitudes.
    This is a classical quantum-inspired optimizer, not quantum hardware.
    Objective: validation MAE, identical trial epochs across candidates.
    """
    seed_everything(int(base_cfg.get('seed',20261109)))
    rng=np.random.default_rng(int(base_cfg.get('seed',20261109)))
    names=['hidden','lr','dropout','gnn_layers']
    low=np.array([16,1.e-4,.05,1.],float);high=np.array([96,3.e-3,.40,3.],float)
    def decode(theta):
        vals=low+(high-low)*np.sin(theta)**2
        return {'hidden':int(np.clip(round(vals[0]/8)*8,16,96)), 'lr':float(vals[1]), 'dropout':float(vals[2]), 'gnn_layers':int(round(vals[3]))}
    def score(theta,k):
        cfg={**base_cfg,**decode(theta),'epochs':trial_epochs,'patience':trial_epochs,'seed':int(base_cfg.get('seed',20261109))+k}
        model,mae=train_central(train,val,scaler,cfg,Path(out_dir)/'qde_trials',device,label=f'trial_{k:03d}')
        del model
        return mae,cfg
    pop=rng.uniform(.12,np.pi/2-.12,size=(population,len(names)))
    costs=[];log=[];count=0
    for i in range(population):
        score_mae,cfg=score(pop[i],count);count+=1;costs.append(score_mae);log.append(dict(eval=count,generation=0,validation_mae_ntu=score_mae,**{k:cfg[k] for k in names}))
    costs=np.array(costs)
    for gen in range(1,generations+1):
        for i in range(population):
            choices=[j for j in range(population) if j!=i]
            r1,r2=rng.choice(choices,2,replace=False)
            best_theta=pop[int(np.argmin(costs))].copy()
            candidate=np.clip(best_theta+.6*(pop[r1]-pop[r2]),.001,np.pi/2-.001)
            mask=rng.random(len(names))<.8;mask[rng.integers(len(names))]=True
            candidate=np.where(mask,candidate,pop[i]);candidate=np.clip(candidate+.04*(best_theta-candidate),.001,np.pi/2-.001)
            score_mae,cfg=score(candidate,count);count+=1
            log.append(dict(eval=count,generation=gen,validation_mae_ntu=score_mae,**{k:cfg[k] for k in names}))
            if score_mae<costs[i]:pop[i]=candidate;costs[i]=score_mae
    best=decode(pop[int(np.argmin(costs))]);Path(out_dir).mkdir(parents=True,exist_ok=True)
    pd.DataFrame(log).to_csv(Path(out_dir)/'qde_search.csv',index=False)
    with open(Path(out_dir)/'qde_best_config.json','w') as f:json.dump(best,f,indent=2)
    return best
