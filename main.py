import urllib.request
import os.path
import tiktoken
import json

# Import sample text and read it

url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt")

file_path = "the_verdict.txt"
txt = ""
with open(file_path) as f: 
  txt = f.read()
  

if not os.path.isfile(file_path) :
  urllib.request.urlretrieve(url, file_path) 

# Encode sample file
encoding = tiktoken.encoding_for_model("gpt2")

enc_path = "encoded.json"
enc = ()
if not os.path.isfile(enc_path):
  enc = encoding.encode(txt)
  with open(enc_path, 'w+') as enc_f:
    enc_f.write(json.dumps(enc))
    enc_f.close()
else:
  with open(enc_path) as enc_f:
    enc = json.loads(enc_f.read())

num_tokens = len(enc)
print(f'Number of tokens : {num_tokens} ')