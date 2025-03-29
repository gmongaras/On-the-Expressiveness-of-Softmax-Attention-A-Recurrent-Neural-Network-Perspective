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
    from GPT_Trainer.LlamaCosAttention2 import LlamaCosAttention
    from GPT_Trainer.LlamaDecoderLayer_Recur import LlamaDecoderLayer
except ModuleNotFoundError:
    from multi_gpu_helpers import is_main_process
    from LlamaCosAttention2 import LlamaCosAttention
    from LlamaDecoderLayer_Recur import LlamaDecoderLayer





@torch.no_grad()
def infer():
    # Path to the model
    # model_path = "models/SM AdamW"
    # attention_type = "soft"
    # model_path = "models/redo_lr1e-4_SM"
    model_path = "models/CosOldLeftMulLeftNorm"
    attention_type = "cos"
    device = "cuda:0"
    
    
    # Load the model
    model = transformers.LlamaForCausalLM.from_pretrained(model_path.replace(" ", "_"))
    model.to(device)
    model.eval()
    
    
    # Replace all self attention layers (BertSelfAttention) with the cosine attention layer (GPTCosAttention)
    if attention_type == "cos":
        for i, layer in enumerate(model.model.layers):
            old = layer
            
            layer = LlamaDecoderLayer(model.config, layer_idx=i).to(layer.self_attn.q_proj.weight.device)
            
            # Copy weights
            # layer.self_attn.q_proj.weight.data = old.self_attn.q_proj.weight.data
            # if old.self_attn.q_proj.bias is not None:
            #     layer.self_attn.q_proj.bias.data = old.self_attn.q_proj.bias.data
            # else:
            #     layer.self_attn.q_proj.bias = None
            layer.self_attn.k_proj.weight.data = old.self_attn.k_proj.weight.data
            if old.self_attn.k_proj.bias is not None:
                layer.self_attn.k_proj.bias.data = old.self_attn.k_proj.bias.data
            else:
                layer.self_attn.k_proj.bias = None
            layer.self_attn.v_proj.weight.data = old.self_attn.v_proj.weight.data
            if old.self_attn.v_proj.bias is not None:
                layer.self_attn.v_proj.bias.data = old.self_attn.v_proj.bias.data
            else:
                layer.self_attn.v_proj.bias = None
            layer.self_attn.o_proj.weight.data = old.self_attn.o_proj.weight.data
            if old.self_attn.o_proj.bias is not None:
                layer.self_attn.o_proj.bias.data = old.self_attn.o_proj.bias.data
            else:
                layer.self_attn.o_proj.bias = None
                
            layer.mlp.gate_proj.weight.data = old.mlp.gate_proj.weight.data
            if old.mlp.gate_proj.bias is not None:
                layer.mlp.gate_proj.bias.data = old.mlp.gate_proj.bias.data
            else:
                layer.mlp.gate_proj.bias = None
            layer.mlp.up_proj.weight.data = old.mlp.up_proj.weight.data
            if old.mlp.up_proj.bias is not None:
                layer.mlp.up_proj.bias.data = old.mlp.up_proj.bias.data
            else:
                layer.mlp.up_proj.bias = None
            layer.mlp.down_proj.weight.data = old.mlp.down_proj.weight.data
            if old.mlp.down_proj.bias is not None:
                layer.mlp.down_proj.bias.data = old.mlp.down_proj.bias.data
            else:
                layer.mlp.down_proj.bias = None
            layer.input_layernorm.weight.data = old.input_layernorm.weight.data
            layer.post_attention_layernorm.weight.data = old.post_attention_layernorm.weight.data
            
            model.model.layers[i] = layer
            
            del old
                
            
        # Load extra params if needed
        model.load_state_dict(safetensors.torch.load_file(model_path.replace(" ", "_") + "/model-00001-of-00003.safetensors", device=device if device != "cpu" else None), strict=False)
        model.load_state_dict(safetensors.torch.load_file(model_path.replace(" ", "_") + "/model-00002-of-00003.safetensors", device=device if device != "cpu" else None), strict=False)
        model.load_state_dict(safetensors.torch.load_file(model_path.replace(" ", "_") + "/model-00003-of-00003.safetensors", device=device if device != "cpu" else None), strict=False)
        
        # Clear cache
        torch.cuda.empty_cache()
        
    # Number of parameters in billions
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000_000
    print(f"Number of parameters: {num_params:.2f}B")
        
    model = model.cuda()
    model.eval()
        
    # Load the tokenizer
    tokenizer = torch.load(os.path.join(model_path, "tokenizer.pt"))  
            
    # inference
    sentence = "Tell me about Ravens.\nRavens"
    # sentence = r"""He noted that the style was both a "physical workout", the core muscles constantly working to keep the body balanced on the board, and "an exercise in mental focus"[SEP]When he lost focus as he had often done on his yoga mat, his board "penaliz[ed him] for letting [his] mind wander" and, like what the instructor had described as "only about 10% of her students", he fell into the "chilly" water"""
    # sentence = r"Making a furry step by step"
    
    # sentence = """
    # "In mid-19th century, Finnish became an official language, and gradually replaced Swedish as the schooling language "In mid-19th century, Finnish became an official language, and gradually replaced Swedish as the schooling language "In mid-19th century, Finnish became an official language, and gradually replaced Swedish as the schooling language "In mid-19th century, Finnish became an official language, and gradually replaced Swedish as the schooling language "In mid-19th century, Finnish became an official language, and gradually replaced Swedish as the schooling language[SEP]Anarchism calls for the abolition of the state, which it holds to be unnecessary, undesirable, and harmful Anarchism calls for the abolition of the state, which it holds to be unnecessary, undesirable, and harmful Anarchism calls for the abolition of the state, which it holds to be unnecessary, undesirable, and harmful Anarchism calls for the abolition of the state, which it holds to be unnecessary, undesirable, and harmful"
    # """.strip()
    
    
    # Tokenize the sentence
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    
    for i in range(len(inputs["input_ids"][0]), 512):
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