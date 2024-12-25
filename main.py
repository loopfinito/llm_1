import urllib.request
import os.path
import tiktoken

# Import sample text and read it

url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt")

file_path = "the_verdict.txt"
txt = ""
with open(file_path) as f: 
  txt = f.read()
  

if not os.path.isfile(file_path) :
  urllib.request.urlretrieve(url, file_path) 

# Encode sample file
encoding = tiktoken.encoding_for_model("gpt-4o")
num_tokens = len(encoding.encode(txt))
print(f'Number of tokens : {num_tokens} ')