"""Local HTTP routing and JSON shapes, using deterministic fixture logits."""
from http.server import HTTPServer
import json
from pathlib import Path
import threading
import unittest
import urllib.error
import urllib.request
import torch
from scripts.serve import handler_for
from scripts.test_contract import Tokenizer

class FixtureModel:
    def eval(self):return self
    def __call__(self,paths,chunk,recompute=False):return torch.arange(len(paths),dtype=torch.float32)

class HTTPTest(unittest.TestCase):
    def test_exact_endpoint_and_typed_answers(self):
        server=HTTPServer(('127.0.0.1',0),handler_for(FixtureModel(),Tokenizer(),{'output_model':'openjev-preview'}))
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        url='http://127.0.0.1:'+str(server.server_port)
        try:
            request=json.loads(Path('examples/jev_request.json').read_text())
            def post(path,value):
                body=json.dumps(value).encode()
                return urllib.request.urlopen(urllib.request.Request(url+path,data=body,headers={'Content-Type':'application/json'}),timeout=5)
            with post('/v1/systemone',request) as response:body=json.load(response)
            self.assertEqual(set(body),{'model','answers','usage'})
            self.assertEqual(set(body['answers']),set(request['questions']))
            for path,req,status in [('/wrong',request,404),('/v1/systemone',{'state':'missing questions'},400),
                                    ('/v1/systemone',{**request,'model':'jev-latest'},400)]:
                with self.assertRaises(urllib.error.HTTPError) as error:post(path,req)
                self.assertEqual(error.exception.code,status)
        finally:
            server.shutdown();thread.join();server.server_close()

if __name__=='__main__':unittest.main()
