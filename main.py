from os import getenv

import classla
import torch
import uvicorn as uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

from src.inference import prepare_sample, cls, gen, df2json, refresh_stats, get_long_words, save_feedback


class Simplify(BaseModel):
    text: str


class RefreshStats(BaseModel):
    text: str


class LongWords(BaseModel):
    text: str
    length: int


class Likes(BaseModel):
    text: str
    value: str


app = FastAPI()

# import models
classla_resources = getenv("CLASSLA_RESOURCES") or "./models/classla_resources"
nlp = classla.Pipeline('sl', dir=classla_resources)

cls_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
cls_handle = getenv("CLS_MODEL_PATH") or "./models/sloberta_slokit_maxlen65_10e_lr2e-05/checkpoint-474/"
cls_tokenizer = AutoTokenizer.from_pretrained(cls_handle, local_files_only=True)
cls_model = AutoModelForSequenceClassification.from_pretrained(cls_handle, local_files_only=True).to(cls_device)
cls_model.eval()
print('CLS device:', cls_model.device)

gen_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
gen_name = getenv("GEN_MODEL_PATH") or "./models/t5-sl-large-v4-maxlen128/checkpoint-936"
gen_tokenizer = AutoTokenizer.from_pretrained(gen_name, local_files_only=True)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_name, local_files_only=True).to(gen_device)
print('GEN device:', gen_model.device)

# import lists
with open('data/frequency-list-from-textbook-corpus-diachronic-lemmas.txt', mode='r', encoding='utf-8') as f:
    textbook = [line.strip() for line in f.readlines()]

with open('data/SloveneFrequentCommonWords-lemmas.txt', mode='r', encoding='utf-8') as f:
    general = [line.strip() for line in f.readlines()]


@app.post("/simplify/")
async def generate_simplification(item: Simplify):
    sample_df = prepare_sample(nlp, item.text)
    sample_df = cls(cls_tokenizer, cls_model, cls_device, sample_df)
    sample_df = gen(gen_tokenizer, gen_model, sample_df)
    json_output = df2json(nlp, sample_df, textbook, general)

    return json_output


@app.post("/refresh-stats/")
async def refresh(item: RefreshStats):
    json_output = refresh_stats(nlp, item.text, textbook, general)

    return json_output


@app.post("/long-words/")
async def long_words(item: LongWords):
    json_output = get_long_words(nlp, item.text, item.length)

    return json_output


@app.post("/likes/")
async def save_likes(item: Likes):
    save_feedback(item.text, item.value)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
