import torch
from transformers import AutoTokenizer, BioGptForCausalLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

inputs = tokenizer("The patient is hospitalized with ischemic stroke and currently has midline shift of 3 mm. Their blood glucose is high, heart rate is very high, blood urea nitrogen is low, and they have not been scanned in a long time. On a scale of 0 mm to 15 mm, the patient's midline shift tomorrow is expected to be", return_tensors="pt")

with torch.no_grad():
    output = model.generate(**inputs, 
                            min_new_tokens = 5, 
                            max_new_tokens = 20, 
                            num_beams = 3,
                            early_stopping = True
                            )
    
print(tokenizer.decode(output[0], skip_special_tokens=True))

