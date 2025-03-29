import transformers
from datasets import load_dataset
import os
import re
import datasets
import tqdm

from concurrent.futures import ProcessPoolExecutor
import itertools
from multiprocessing import cpu_count
import logging

TOKEN = "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"





def main():
    # Cache dirs
    cache_path = "PILECACHE/"
    tok_cache_path = "PILECACHE/"
    
    # Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True, token=TOKEN, cache_dir="GPT_Trainer/llama2")
    
    # Load in datasets
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    dataset = datasets.load_dataset(f"gmongaras/EleutherAI_the_pile_deduplicated", cache_dir=cache_path)["train"]
    
    # Tokenize the data
    if not os.path.exists(tok_cache_path):
        os.makedirs(tok_cache_path)
    dataset = dataset.map(tokenizer, batched=True, batch_size=1000, num_proc=16, input_columns=["text"], remove_columns=["text"], cache_file_name=tok_cache_path + "/tokenized_dataset.arrow")

    # Push to hub
    dataset.push_to_hub(f"gmongaras/Pile_Llama_Tokenized", token=TOKEN)
    
    
    
    
if __name__ == "__main__":
    main()
