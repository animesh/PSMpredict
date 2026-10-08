from __future__ import annotations
import csv, json, math
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from .core import *

def train(data_path, output_dir, cfg: Config, resume=None):
    seed_all(cfg.seed); out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    records = torch.load(data_path, weights_only=False); n = len(records); indices = torch.randperm(n, generator=torch.Generator().manual_seed(cfg.seed)).tolist(); nv=max(1,int(n*cfg.val_fraction)); va, tr=indices[:nv],indices[nv:]
    train_loader=DataLoader(Subset(SpectrumDataset(records),tr),batch_size=cfg.batch_size,shuffle=True,num_workers=cfg.num_workers,collate_fn=collate,pin_memory=True)
    val_loader=DataLoader(Subset(SpectrumDataset(records),va),batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,collate_fn=collate,pin_memory=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=SpectrumTransformer(cfg).to(device)
    if resume: model.load_state_dict(torch.load(resume,map_location=device,weights_only=False)["model"])
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay); scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    best=-1.; history=[]
    for epoch in range(1,cfg.epochs+1):
        model.train(); total=count=0
        for aa,mods,charge,pad,target,cmask in train_loader:
            aa,mods,charge,pad,target,cmask=[x.to(device,non_blocking=True) for x in (aa,mods,charge,pad,target,cmask)]; opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
                pred=model(aa,mods,charge,pad); loss=torch.nn.functional.mse_loss(pred[cmask],target[cmask])
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); total+=loss.item()*aa.size(0); count+=aa.size(0)
        model.eval(); angles=[]
        with torch.no_grad():
            for aa,mods,charge,pad,target,cmask in val_loader:
                aa,mods,charge,pad,target,cmask=[x.to(device) for x in (aa,mods,charge,pad,target,cmask)]; angles.extend(spectral_angle(model(aa,mods,charge,pad),target,cmask).cpu().tolist())
        metric=float(sum(angles)/len(angles)); row={"epoch":epoch,"train_mse":total/count,"val_spectral_angle":metric}; history.append(row); print(json.dumps(row),flush=True)
        save_checkpoint(out/"last.pt",model,cfg,row)
        if metric>best: best=metric; save_checkpoint(out/"best.pt",model,cfg,row)
    pd.DataFrame(history).to_csv(out/"history.tsv",sep="\t",index=False); return {"best_val_spectral_angle":best,"device":str(device),"train_psms":len(tr),"val_psms":len(va)}

def fragment_mz(sequence, mod_mass):
    residue=[AA_MASS[a]+float(mod_mass[i]) for i,a in enumerate(sequence)]; prefix=[]; s=0
    for x in residue[:-1]: s+=x; prefix.append(s)
    suffix=[]; s=0
    for x in residue[:0:-1]: s+=x; suffix.append(s)
    out={}
    for cut in range(len(sequence)-1):
        bmass=prefix[cut]; ymass=suffix[len(sequence)-cut-2]+H2O
        for series,mass,ordn in (("b",bmass,cut+1),("y",ymass,len(sequence)-cut-1)):
            for z in (1,2):
                for loss,delta in (("",0.),("-H2O",H2O),("-NH3",NH3)):
                    out[(cut,ION_TO_ID[f"{series}{z}{loss}"])]=(f"{series}{ordn}"+(f"({z}+)" if z>1 else "")+loss,(mass-delta+z*PROTON)/z)
    return out

def predict_file(checkpoint, input_tsv, output_tsv, batch_size=512):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model,cfg,_=load_checkpoint(checkpoint,device)
    df=pd.read_csv(input_tsv,sep="\t"); seq_col="Sequence" if "Sequence" in df else "sequence"; charge_col="Charge" if "Charge" in df else "charge"; mod_col="Modified sequence" if "Modified sequence" in df else None
    records=[]
    for i,row in df.iterrows():
        seq=clean_sequence(row[seq_col]); modified=row[mod_col] if mod_col else seq
        if 2<=len(seq)<=cfg.max_len and all(a in AA_TO_ID for a in seq): records.append({"sequence":seq,"modified_sequence":str(modified),"charge":int(row.get(charge_col,2)),"mod_mass":modification_vector(seq,modified),"row_id":i})
    loader=DataLoader(SpectrumDataset(records),batch_size=batch_size,shuffle=False,collate_fn=collate); output=[]; cursor=0
    with torch.no_grad():
        for aa,mods,charge,pad,_,cmask in loader:
            pred=model(aa.to(device),mods.to(device),charge.to(device),pad.to(device)).float().cpu().numpy()
            for j in range(len(aa)):
                r=records[cursor]; n=len(r["sequence"]); mzmap=fragment_mz(r["sequence"],r["mod_mass"])
                for cut in range(n-1):
                    for chan in range(len(ION_TYPES)):
                        label,mz=mzmap[(cut,chan)]; output.append((r["row_id"],r["sequence"],r["charge"],label,mz,float(pred[j,cut,chan])))
                cursor+=1
    pd.DataFrame(output,columns=["row_id","sequence","precursor_charge","fragment","mz","predicted_intensity"]).to_csv(output_tsv,sep="\t",index=False)
    return {"peptides":len(records),"fragments":len(output),"device":str(device)}
