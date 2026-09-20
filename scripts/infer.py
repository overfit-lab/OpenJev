"""Read one Jev-shaped JSON request; return all three typed answers as JSON."""
import argparse
import json
from pathlib import Path
import sys
import torch
try:
    from .model import load_checkpoint, predict
except ImportError:
    from model import load_checkpoint, predict

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--input',type=Path,help='JSON file; otherwise read stdin')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--candidate-chunk',type=int,default=2)
    args=parser.parse_args()
    request=json.loads(args.input.read_text()) if args.input else json.load(sys.stdin)
    dtype=torch.bfloat16 if args.device.startswith('cuda') else torch.float32
    model,_,tokenizer,metadata=load_checkpoint(args.checkpoint,args.device,dtype)
    print(json.dumps(predict(model,tokenizer,metadata,request,args.candidate_chunk),ensure_ascii=False,allow_nan=False))

if __name__=='__main__':main()
