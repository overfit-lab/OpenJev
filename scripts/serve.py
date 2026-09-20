"""Local reference HTTP API for Jev-shaped requests; sequential model execution."""
import argparse
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
from pathlib import Path
import torch
try:
    from .model import load_checkpoint,predict
except ImportError:
    from model import load_checkpoint,predict


def handler_for(model,tokenizer,metadata,chunk_size=2):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass  # No raw user payloads or local paths in access logs.
        def send_json(self,status,payload):
            data=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_POST(self):
            if self.path!='/v1/systemone':return self.send_json(404,{'error':{'code':'not_found'}})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=2_000_000:raise ValueError('Request body must be 1–2000000 bytes')
                request=json.loads(self.rfile.read(length))
                result=predict(model,tokenizer,metadata,request,chunk_size)
            except (ValueError,TypeError,KeyError,UnicodeDecodeError):
                return self.send_json(400,{'error':{'code':'invalid_request'}})
            except Exception:
                return self.send_json(500,{'error':{'code':'inference_failed'}})
            return self.send_json(200,result)
    return Handler


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--device',default='cpu');parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--candidate-chunk',type=int,default=2)
    args=parser.parse_args()
    dtype=torch.bfloat16 if args.device.startswith('cuda') else torch.float32
    model,_,tokenizer,metadata=load_checkpoint(args.checkpoint,args.device,dtype)
    server=HTTPServer(('127.0.0.1',args.port),handler_for(model,tokenizer,metadata,args.candidate_chunk))
    server.serve_forever()

if __name__=='__main__':main()
