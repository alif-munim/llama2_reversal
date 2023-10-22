import json
import pandas as pd

with open('../data/reverse_experiments/june_version_7921032488/all_prompts_train.jsonl') as f:
    train_df = pd.DataFrame(json.loads(line) for line in f).rename(columns={"prompt": "question", "completion": "answer"})
    
with open('../data/reverse_experiments/june_version_7921032488/validation_prompts.jsonl') as f:
    val_df = pd.DataFrame(json.loads(line) for line in f).rename(columns={"prompt": "question", "completion": "answer"})
    
train_df['answer'] = train_df['answer'].apply(lambda x: [x])
train_df['id'] = train_df.question.map(hash)
train_df['id'] = train_df['id'].map(str)
train_df = train_df.reindex(columns=['id', 'question', 'answer'])
train_dict = train_df.to_dict('records')

val_df['answer'] = val_df['answer'].apply(lambda x: [x])
val_df['id'] = val_df.question.map(hash)
val_df['id'] = val_df['id'].map(str)
val_df = val_df.reindex(columns=['id', 'question', 'answer'])
val_dict = val_df.to_dict('records')

# print(f'validation: {val_dict[:10]}\n\ntrain: {train_dict[:10]}')
# print(len(val_dict))
# print(len(train_dict))

with open('data/revcurse-train.json', 'w') as fout:
    json.dump(train_dict, fout)
    
with open('data/revcurse-val.json', 'w') as fout:
    json.dump(val_dict, fout)