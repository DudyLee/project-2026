"""Small mathematical/split checks; no experiment training or private inputs."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import methods as M


def sample():
    rows=[]
    for i,room in enumerate(M.STUDY['development_rooms']):
        for angle in M.STUDY['angles']:
            rows.append(dict(Room_ID=room,Room_Area_m2=20+i,Glass_Area_m2=3,
                Orientation_X_model=0,Orientation_Y_model=1,Solar_Obstruction_Ratio=.2,
                DOY_1based=100,Hour=12,PVSD_Angle_deg=angle,EPW_GHI_Wh_m2=400,
                EPW_DNI_Wh_m2=600,EPW_DHI_Wh_m2=80,Solar_Altitude_mid_deg=30,
                Solar_Azimuth_mid_deg=90,Annual_Incident_Shaded_kWh_m2=150,
                E_mean_lux=100+i,E_P10_lux=5+i,F_300_fraction=.2,F_UDI_fraction=.6,PV_AC_kWh=.03))
    return pd.DataFrame(rows)


class MethodsTest(unittest.TestCase):
    def test_feature_direction_and_no_identifiers(self):
        x,q=M.features(sample())
        self.assertEqual([x[k].shape[1] for k in ['F0','F1','F2','B']],[15,21,24,30])
        np.testing.assert_allclose(x['F1'].DNI_u,600*np.cos(np.pi/6))
        np.testing.assert_allclose(x['F1'].DNI_v,0,atol=1e-12)
        for f in x.values():self.assertFalse(set(f)&set(['Room_ID']+M.TARGETS))
        self.assertTrue((q>0).all())

    def test_target_transforms_round_trip(self):
        d=sample();_,q=M.features(d)
        for v in range(8,12):
            for m,s in M.specs(v).items():
                if s['family']=='frozen_tree':continue
                targets=['E_P10_lux'] if m=='S1_direct' else M.TARGETS
                names,y=M.labels(d,q,v,m,s,targets);p=M.restore(y,names,q,v,m,s,targets)
                for t in targets:np.testing.assert_allclose(p[t][0],d[t],rtol=1e-12,atol=1e-12)

    def test_group_boundaries(self):
        d=sample();jobs,protocols=M.outer_splits()
        self.assertEqual(len(jobs),9);self.assertEqual(len(protocols),10)
        for job in jobs:
            fit=set(job['fit_rooms']);held=set(job['held_rooms']);self.assertFalse(fit&held)
            self.assertFalse(fit&set(M.STUDY['excluded_test_rooms']))
            for stable in [False,True]:
                for s in M.inner_splits(d,job['fit_rooms'],stable):
                    self.assertTrue(set(s['fit_rooms']+s['held_rooms'])<=fit)
                    self.assertFalse(set(s['fit_rooms'])&set(s['held_rooms']))
        for p in protocols:
            if p['fold']=='posX':self.assertEqual(p['job_id'],'room_BM223')

    def test_ridge_targets_are_independent(self):
        rng=np.random.default_rng(2026);x=rng.normal(size=(50,5));y=rng.normal(size=(50,3))
        combined=Ridge(alpha=5,solver='svd').fit(x,y).predict(x)
        for j in range(3):np.testing.assert_allclose(combined[:,j],Ridge(alpha=5,solver='svd').fit(x,y[:,j]).predict(x),rtol=1e-12,atol=1e-12)

    def test_fusion_after_branch_restore(self):
        s=M.specs(11)['A0'];sq=M.specs(11)['Aq']
        p=M.restore(np.array([[-2.],[2.]]),['F_UDI_fraction'],np.array([.5,2.]),11,'A0',s,['F_UDI_fraction'])
        pq=M.restore(np.array([[2.],[-2.]]),['F_UDI_fraction'],np.array([.5,2.]),11,'Aq',sq,['F_UDI_fraction'])
        pred={'A0':p,'Aq':pq,'B0':p,'Bq':pq};M.add_fusion(pred)
        np.testing.assert_array_equal(pred['CA']['F_UDI_fraction'][0],[.5,.5])


if __name__=='__main__':unittest.main()
