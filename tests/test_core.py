import numpy as np, pandas as pd
from psmpredict.core import *
def test_mapping():
    r=pd.Series({"Sequence":"PEPTIDE","Modified sequence":"_PEPTIDE_","Charge":2,"Matches":"b2;y3(2+);b3-H2O","Intensities":"10;20;5"})
    x=row_to_record(r,60); assert x["target"].shape==(6,12); assert x["target"].max()==1
def test_model():
    cfg=Config(max_len=10,d_model=24,nhead=4,layers=1)
    model=SpectrumTransformer(cfg); aa=torch.tensor([[AA_TO_ID[x] for x in "PEPTIDE"]]); mods=torch.zeros_like(aa,dtype=torch.float); charge=torch.tensor([2]); mask=torch.zeros_like(aa,dtype=torch.bool)
    assert model(aa,mods,charge,mask).shape==(1,6,12)
