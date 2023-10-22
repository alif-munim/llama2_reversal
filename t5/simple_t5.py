from simplet5 import SimpleT5
from datasets import load_dataset
import pandas as pd
import json

# train_ds = load_dataset("lberglund/reversal_curse", split="train")
# val_ds = load_dataset("lberglund/reversal_curse", split="train")

# train_df = train_ds.to_pandas()
# val_df = val_ds.to_pandas()

# print(len(train_df))

with open('data/reverse_experiments/june_version_7921032488/all_prompts_train.jsonl') as f:
    train_df = pd.DataFrame(json.loads(line) for line in f).rename(columns={"prompt": "source_text", "completion": "target_text"})
    
with open('data/reverse_experiments/june_version_7921032488/validation_prompts.jsonl') as f:
    val_df = pd.DataFrame(json.loads(line) for line in f).rename(columns={"prompt": "source_text", "completion": "target_text"})

model = SimpleT5()
model.from_pretrained(model_type="t5", model_name="t5-base")
model.load_model('outputs/simplet5-epoch-2-train-loss-0.0592-val-loss-10.9152')

model.train(train_df = train_df,
    eval_df = val_df, 
    source_max_token_len=128, 
    target_max_token_len=50, 
    batch_size=64, max_epochs=100, use_gpu=True)

