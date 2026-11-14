import unittest
import numpy as np
import torch
from src.hydrograph import HydroGraphFL,load_arrays,physics_flux_loss
class TestHydroGraph(unittest.TestCase):
 def test_graph_output_and_backward(self):
  m=HydroGraphFL(hidden=16,layers=1,gnn_layers=2,dropout=.1)
  x=torch.randn(2,7,14,4);a=torch.zeros(2,7,7)
  for i in range(1,7):a[:,i,i-1]=1
  y=m(x,a);self.assertEqual(y.shape,(2,7));y.mean().backward()
  self.assertIsNotNone(m.gru.weight_ih_l0.grad)
 def test_missing_physics_terms(self):
  v=torch.randn(2,7,requires_grad=True)
  zero,n=physics_flux_loss(v,None)
  self.assertEqual(n,0);self.assertEqual(zero.item(),0)
 def test_dataset(self):
  rows,stats=load_arrays('data/benchmark',window=14)
  self.assertEqual(len([s for s in rows if s['split']=='test'])*7,2940)
  self.assertEqual(len({r['client'] for r in rows}),5)
if __name__=='__main__':unittest.main()
