import numpy as np

text = 'ID: 18, Sex: F, Age: 52 y --- AIS: B - NLI: C5 --- 10 days after trauma, (session:1)'
print(f"text: {text}")
text = text.split()
print(f"text: {text}")
id_pt = " ".join(text[:7])
print(f"{id_pt}")
ais_nli = "-".join([text[9],text[12]])
print(f"{ais_nli}")
days = text[14]
print(f"days: {days}")


