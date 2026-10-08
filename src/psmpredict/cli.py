import argparse, json
from .core import Config, prepare_msms
from .workflows import train, predict_file

def add_cfg(p):
    for name,typ,default in [("max_len",int,60),("d_model",int,192),("nhead",int,6),("layers",int,4),("dropout",float,.1),("batch_size",int,256),("epochs",int,15),("learning_rate",float,2e-4),("weight_decay",float,1e-4),("seed",int,42),("val_fraction",float,.1),("num_workers",int,4),("max_psms",int,None)]: p.add_argument("--"+name.replace("_","-"),type=typ,default=default)
def cfg(a): return Config(**{k:getattr(a,k) for k in Config.__dataclass_fields__ if hasattr(a,k)})
def main():
    ap=argparse.ArgumentParser(prog="psmpredict"); sub=ap.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("prepare"); p.add_argument("inputs",nargs="+"); p.add_argument("--output",required=True); add_cfg(p); p.add_argument("--min-score",type=float)
    t=sub.add_parser("train"); t.add_argument("--data",required=True); t.add_argument("--output-dir",required=True); t.add_argument("--resume"); add_cfg(t)
    q=sub.add_parser("predict"); q.add_argument("--checkpoint",required=True); q.add_argument("--input",required=True); q.add_argument("--output",required=True); q.add_argument("--batch-size",type=int,default=512)
    a=ap.parse_args()
    if a.cmd=="prepare": result=prepare_msms(a.inputs,a.output,cfg(a))
    elif a.cmd=="train": result=train(a.data,a.output_dir,cfg(a),a.resume)
    else: result=predict_file(a.checkpoint,a.input,a.output,a.batch_size)
    print(json.dumps(result,indent=2))
if __name__=="__main__": main()
