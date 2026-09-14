import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
model = AutoModelForMaskedLM.from_pretrained("yikuan8/Clinical-Longformer")

text = "The patient is hospitalized with ischemic stroke. The patient currently has moderate midline shift. Their blood glucose is high, heart rate is very high, and they have not been scanned in a long time. Tomorrow, the patient's midline shift will be <mask>"
input_ids = tokenizer([text], return_tensors="pt")["input_ids"]

print("done")

