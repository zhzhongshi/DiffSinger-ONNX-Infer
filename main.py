import importlib
import re
import os.path
import uuid 
import uvicorn

from pathlib import *
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI
from fastapi.responses import FileResponse
from infer import Infer
from utils.hparams import set_hparams
from utils.hparams import hparams as hp
import numpy as np
import soundfile as sf

app = FastAPI()

class MainInfer:
    def __init__(self, inference_cls="infer.Infer"):
        pkg = ".".join(inference_cls.split(".")[:-1])
        cls_name = inference_cls.split(".")[-1]
        self.inference_cls = getattr(importlib.import_module(pkg), cls_name)

    def greet(self, text: str, notes:str, notes_duration:str):
        PUNCS = '。？；：'
        sents = re.split(rf'([{PUNCS}])', text.replace('\n', ','))
        sents_notes = re.split(rf'([{PUNCS}])', notes.replace('\n', ','))
        sents_notes_dur = re.split(rf'([{PUNCS}])', notes_duration.replace('\n', ','))

        if sents[-1] not in list(PUNCS):
            sents = sents + ['']
            sents_notes = sents_notes + ['']
            sents_notes_dur = sents_notes_dur + ['']

        audio_outs = []
        s, n, n_dur = "", "", ""
        for i in range(0, len(sents), 2):
            if len(sents[i]) > 0:
                s += sents[i] + sents[i + 1]
                n += sents_notes[i] + sents_notes[i+1]
                n_dur += sents_notes_dur[i] + sents_notes_dur[i+1]
            if len(s) >= 400 or (i >= len(sents) - 2 and len(s) > 0):
                audio_out = self.infer_ins.infer_once({
                    'text': s,
                    'notes': n,
                    'notes_duration': n_dur,
                })
                audio_out = audio_out * 32767
                audio_out = audio_out.astype(np.int16)
                audio_outs.append(audio_out)
                audio_outs.append(np.zeros(int(hp['audio_sample_rate'] * 0.3)).astype(np.int16))
                s = ""
                n = ""
        audio_outs = np.concatenate(audio_outs)
        return hp['audio_sample_rate'], audio_outs

    def run(self,text,note,duration):
        set_hparams("model/config.yaml")
        infer_cls = self.inference_cls
        self.infer_ins: Infer = infer_cls(hp)
        sr, wav = self.greet(text, note, duration)
        return sr,wav
        
class DataModel(BaseModel):
    data: List[str]
@app.post("/api/infer/")
def inference(item: DataModel):
    input=item.dict()
    text=input["data"][0]
    note=input["data"][1]
    duration=input["data"][2]
    g = MainInfer()
    sr,wav=g.run(text,note,duration)
    uuid1=uuid.uuid1()
    if Path("cache/").exists():
        sf.write(file=f'cache/{uuid1}.wav', data=wav, samplerate=sr)
    else:
        Path.mkdir(Path("cache/"))
        sf.write(file=f'cache/{uuid1}.wav', data=wav, samplerate=sr)
    return {"id":uuid1}

@app.get("/api/getwav/",response_class=FileResponse)
def getwav(uuid: str):
    print(uuid)
    return FileResponse(f"cache/{uuid}.wav",media_type="audio/wav")

if __name__ == '__main__':
    uvicorn.run(app)
