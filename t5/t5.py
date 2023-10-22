from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

t5_qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-11b-ssm-tqa")
t5_tok = AutoTokenizer.from_pretrained("google/t5-11b-ssm-tqa")

# input_ids = t5_tok("When was Franklin D. Roosevelt born?", return_tensors="pt").input_ids
input_ids = t5_tok("1882 is the birth year of former president", return_tensors="pt").input_ids
gen_output = t5_qa_model.generate(input_ids)[0]

print(t5_tok.decode(gen_output, skip_special_tokens=True))
