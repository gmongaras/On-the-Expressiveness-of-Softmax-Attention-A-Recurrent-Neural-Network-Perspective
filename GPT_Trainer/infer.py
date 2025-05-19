import torch
from torch import nn
import transformers
import datasets
import os
import wandb
from tqdm import tqdm
from contextlib import nullcontext
import safetensors


try:
    from GPT_Trainer.multi_gpu_helpers import is_main_process
    from GPT_Trainer.LlamaDecoderLayer import LlamaDecoderLayer
except ModuleNotFoundError:
    from multi_gpu_helpers import is_main_process
    from LlamaDecoderLayer import LlamaDecoderLayer





@torch.no_grad()
def infer():
    # Path to the model
    attention_type = "gated_softmax_plusplus_extratoks"
    model_path = "models/fineweb_gated_softmax_plusplus_extratoks_35bs_2gpu_1024seqlenV2/"
    device = "cuda:0"
    model_max_length = 1024


    # Read token from .env file
    with open(".env", "r") as f:
        token = f.read().strip()

    # Tokenizer
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False, cache_dir="GPT_Trainer/llama2", token=token)
        # self.tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", use_fast=False, cache_dir="GPT_Trainer/gpt-j-6B")
    except OSError:
        raise FileNotFoundError("Token not found in .env file or user does not have access to Llama 2 weights with that token. Please add your Hugging Face token to the .env file.")
    
    tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_token = torch.tensor([tokenizer.pad_token_id])
    
    # Set max sequence length
    tokenizer.model_max_length = model_max_length

    # GPT-J Model. We are training it from scratch
    model = transformers.LlamaForCausalLM(config=transformers.LlamaConfig.from_dict({
        "_name_or_path": "meta-llama/Llama-2-7b-hf",
        "architectures": [
            "LlamaForCausalLM"
        ],
        "bos_token_id": 1,
        "eos_token_id": 2,
        "hidden_act": "silu",
        "hidden_size": 1024, #4096,
        "initializer_range": 0.02,
        "intermediate_size": 1024*2, # 11008
        "max_position_embeddings": model_max_length,
        "model_type": "llama",
        "num_attention_heads": 16,
        "num_hidden_layers": 20,
        "num_key_value_heads": 16,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": None,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "use_cache": True,
        # "vocab_size": 32000,
        "vocab_size": tokenizer.vocab_size,
        "attention_type": attention_type,
    }))

    
    # Replace all self attention layers with the cosine attention layer
    for i, layer in enumerate(model.model.layers):
        old = layer
        model.model.layers[i] = LlamaDecoderLayer(model.config, layer_idx=i).to(layer.self_attn.q_proj.weight.device)
        model.model.layers[i].self_attn.layer_num = i
        del old

    # Load in params
    model.load_state_dict(safetensors.torch.load_file(model_path + "/model.safetensors"), strict=True)
    model.eval()
    
    
    # Clear cache
    torch.cuda.empty_cache()
    
    # Number of parameters in billions
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000_000
    print(f"Number of parameters: {num_params:.2f}B")
        
    model = model.cuda()
    model.eval()
    
    # Load the tokenizer
    # tokenizer = torch.load(os.path.join(model_path, "tokenizer.pt"))  
            
    # inference
    sentence = "Tell me about Ravens.\nRavens"
    
    
    # Tokenize the sentence
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    
    for i in range(len(inputs["input_ids"][0]), model_max_length):
        # Get the logits
        if attention_type == "cos":
            outputs = model(**inputs)
        else:
            outputs = model(**inputs, output_attentions=True)
            
            # for attn in outputs.attentions:
            #     # Matplotlib attention heatmap
            #     import matplotlib.pyplot as plt
            #     probs = attn[0].detach().cpu().numpy()
            #     for head in range(probs.shape[0]):
            #         # Shape is (num_heads, seq_len, seq_len)
            #         plt.imshow(probs[head])
            #         plt.show()
            #         if not os.path.exists("imgs"):
            #             os.makedirs("imgs")
            #         plt.savefig(f"imgs/attention{head}.png")
                    
            #     print()
            
        # Get the predicted next word
        logits = outputs.logits[0, -1]
        # Set prob of <|endoftext|> to 0
        # logits[50256] = -float("inf")
        dist = torch.distributions.Categorical(logits=logits)
        next_word = dist.sample()
        if next_word == 50256:
            break
        
        # Add the next word to the input
        inputs["input_ids"] = torch.cat([inputs["input_ids"], next_word.unsqueeze(0).unsqueeze(0)], dim=1)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones(1, 1).cuda()], dim=1)
        
    # Decode the output
    decoded = tokenizer.decode(inputs["input_ids"][0])
    
    print(decoded)
    
    
    
if __name__ == "__main__":
    infer()