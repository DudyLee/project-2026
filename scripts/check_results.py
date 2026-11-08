"""Validate the small published aggregate tables without raw data or weights."""
from pathlib import Path
import numpy as np
import pandas as pd
import methods as M


def main():
    folder=Path(__file__).resolve().parents[1]/'results'
    summary=pd.read_csv(folder/'summary_metrics.csv')
    rooms=pd.read_csv(folder/'room_metrics.csv')
    assert len(summary)==258 and len(rooms)==1462
    keys=['version','protocol','method','target']
    assert not summary.duplicated(keys).any()
    assert not rooms.duplicated(keys+['room']).any()
    for r in summary.itertuples():
        g=rooms[(rooms.version==r.version)&(rooms.protocol==r.protocol)&(rooms.method==r.method)&(rooms.target==r.target)]
        expected=M.STUDY['excluded_test_rooms'] if r.protocol=='Holdout3Rooms' else M.STUDY['development_rooms']
        assert set(g.room)==set(expected) and len(g)==r.room_count==len(expected)
        assert g.n.eq(M.STUDY['rows_per_room']).all()
        worst=g.sort_values(['mae','room'],ascending=[False,True]).iloc[0]
        np.testing.assert_allclose([g.mae.mean(),g.mae.max(),g.rmse.mean()],[r.macro_MAE,r.worst_MAE,r.macro_RMSE],rtol=1e-12,atol=1e-12)
        assert worst.room==r.worst_room
    hist=pd.read_csv(folder/'history_mae.csv')
    for r in summary.itertuples():
        g=hist[(hist.version==f'v{r.version}+')&(hist.protocol==r.protocol)&(hist.method==r.method)]
        assert len(g)==1
        col,scale={
            'E_mean_lux':('E_mean_MAE_lux',1),'E_P10_lux':('E_P10_MAE_lux',1),
            'F_300_fraction':('F300_MAE_pp',100),'F_UDI_fraction':('FUDI_MAE_pp',100),'PV_AC_kWh':('PV_MAE_kWh',1),
        }[r.target]
        np.testing.assert_allclose(g.iloc[0][col],r.macro_MAE*scale,rtol=1e-12,atol=1e-12)
    print('Verified: 258 summary cells, 1462 room metrics, and historical table units.')


if __name__=='__main__':main()
