"""Execute real neural training on generated-data CSVs, write prediction files and metrics."""
import argparse,json,os
from pathlib import Path
import numpy as np
import pandas as pd
from src.hydrograph import load_arrays,seed_everything,train_central,fed_train,qde_optimize,evaluate

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--mode',choices=['central','fedavg','fedprox','qde','all'],default='central')
 p.add_argument('--quick',action='store_true',help='Fast smoke test with 1 epoch/round')
 p.add_argument('--device',default='cpu')
 p.add_argument('--data-dir',default='data')
 p.add_argument('--output-dir',default='outputs/neural')
 p.add_argument('--config',default='configs/default.json')
 a=p.parse_args()
 cfg=json.load(open(a.config));seed_everything(cfg['seed'])
 if a.quick:cfg.update(epochs=1,rounds=1,local_epochs=1,patience=1,hidden=16,gru_layers=1,gnn_layers=1,batch=32)
 samples,scaler=load_arrays(a.data_dir,window=cfg['window'])
 splits={k:[r for r in samples if r['split']==k] for k in ['train','validation','val','test']}
 train=splits['train']; val=splits['validation'] or splits['val'];test=splits['test']
 if not train or not val or not test:raise ValueError('Train/validation/test partition is empty; inspect CSV split labels')
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 original=pd.read_csv(Path(a.data_dir)/'forecast_supervised_rows.csv')
 threshold=float(np.quantile(original[original.split=='train'].actual_ntu,.75))
 (out/'run_provenance.json').write_text(json.dumps({'data_origin':'computer_generated','seed':cfg['seed'],'train_group_days':len(train),'validation_group_days':len(val),'test_group_days':len(test),'test_station_days':len(test)*7,'threshold_ntu':threshold,'physics_constraints_valid':0,'full_neural_model_executed':True,'site_observations_used':False,'config':cfg},indent=2))
 modes=['central','fedavg','fedprox','qde'] if a.mode=='all' else [a.mode]
 metrics=[]
 for mode in modes:
  selected=cfg.copy()
  if mode=='qde':
   best=qde_optimize(train,val,scaler,cfg,out,device=a.device,population=4,generations=1 if a.quick else int(cfg['qde_generations']),trial_epochs=1 if a.quick else int(cfg['qde_trial_epochs']))
   selected.update(best)
   model,bestval=train_central(train,val,scaler,selected,out,a.device,'qde_tuned_hydrograph_fl')
  elif mode=='central':model,bestval=train_central(train,val,scaler,selected,out,a.device,'central_hydrograph_fl')
  else:model,bestval=fed_train(train,val,scaler,selected,out,a.device,algorithm=mode)
  met,preds=evaluate(model,test,scaler,device=a.device,threshold=threshold)
  met.update(model=mode,validation_mae_ntu=bestval,model_class='GRU-directed-GNN',data_origin='computer_generated',physics_constraints_valid=0)
  preds.to_csv(out/f'{mode}_test_predictions.csv',index=False)
  metrics.append(met)
  print(mode, {k:round(v,5) if isinstance(v,float) else v for k,v in met.items()})
 pd.DataFrame(metrics).to_csv(out/'neural_comparison_metrics.csv',index=False)
 print('Saved to',out.resolve())
if __name__=='__main__':main()
