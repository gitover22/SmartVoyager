import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

base_path = './model/rerank_model/'
# download repo to the base_path directory using git
# os.system('apt install git')
# os.system('apt install git-lfs')
os.system(f'git clone https://code.openxlab.org.cn/answer-qzd/bge_rerank.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')