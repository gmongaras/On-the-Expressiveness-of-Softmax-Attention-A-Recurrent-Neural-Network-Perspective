import torch
from torch import nn
import transformers
import datasets
import os
import wandb
from tqdm import tqdm
from contextlib import nullcontext
import copy
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import safetensors


from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

try:
    from GPT_Trainer.multi_gpu_helpers import is_main_process
    from GPT_Trainer.LlamaDecoderLayer import LlamaDecoderLayer
except ModuleNotFoundError:
    from multi_gpu_helpers import is_main_process
    from LlamaDecoderLayer import LlamaDecoderLayer









def init_distributed():
    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    
    # Try the nccl backend
    try:
        dist.init_process_group(
                backend="nccl",
                init_method=dist_url,
                world_size=world_size,
                device_id=torch.device(f"cuda:{local_rank}"),
                rank=rank)
    # Use the gloo backend if nccl isn't supported
    except RuntimeError:
        dist.init_process_group(
                backend="gloo",
                init_method=dist_url,
                world_size=world_size,
                device_id=torch.device(f"cuda:{local_rank}"),
                rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()














def get_scheduler(optimizer, warmup_steps, total_steps):
    # Define the lambda function for the learning rate schedule
    # this value 
    lr_lambda = lambda step: (
        # Warmup
        step/warmup_steps if step < warmup_steps
        # Decrease from 1 to 0 from warmup_steps to total_steps
        else (1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
    )
    
    # # Instead we can decrease from 1 to a percentage of the original learning rate
    # per = 0.1
    # lr_lambda = lambda step: (
    #     # Warmup
    #     step/warmup_steps if step < warmup_steps
    #     # Decrease from 1 to a percentage of the original learning rate
    #     else (1.0 - (1-per)*(step - warmup_steps) / (total_steps - warmup_steps))
    # )

    # Create the scheduler
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)





class Trainer():
    def __init__(self, 
            dataset,
            model_size="small",
            batch_size=256,
            learning_rate=1e-4,
            warmup_steps=10_000,
            num_steps=1_000_000,
            num_steps_early_stop=100_000, 
            dev="cpu",
            wandb_name=None,
            log_steps=10,
            use_amp=True,
            attention_type="soft",
            mlp_type="gelu",
            clipping_value=None,
            weight_decay=0.01,
            model_save_path=None,
            num_save_steps=10_000,
            keep_dataset_in_mem=False,
            load_checkpoint=False,
            checkpoint_path=None,
            finetune=False,
            finetune_task=None,
            model_max_length=4096,
            test_per=0.1,
            num_steps_test=10_000,
        ):
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.num_steps = num_steps
        self.num_steps_early_stop = num_steps_early_stop
        self.wandb_name = wandb_name
        self.log_steps = log_steps
        self.use_amp = use_amp
        self.dev = dev
        self.clipping_value = clipping_value
        self.weight_decay = weight_decay
        self.model_save_path = model_save_path.replace(" ", "_") if model_save_path is not None else None
        self.num_save_steps = num_save_steps
        self.keep_dataset_in_mem = keep_dataset_in_mem
        self.finetune_ = finetune
        self.finetune_task = finetune_task
        self.test_per = test_per
        self.num_steps_test = num_steps_test
        self.model_max_length = model_max_length

        self.padding_side = "right" if attention_type in ["linear_mamba", "linear_rwkv", "gated_softmax_plusplus"] else "left"
        
        
        
        # Must load a checkpoint if finetuning
        if self.finetune_:
            assert load_checkpoint, "Must load a checkpoint if finetuning"
            assert checkpoint_path is not None, "Must provide a checkpoint path if finetuning"


        
        
        
        
        
        # Divide the batch size by the number of GPUs
        if dev != "cpu":
            batch_size = batch_size // int(os.environ['WORLD_SIZE'])
        else:
            batch_size = batch_size
        self.batch_size = batch_size
            
        # Read token from .env file
        with open(".env", "r") as f:
            token = f.read().strip()
        

        if load_checkpoint:
            self.load_checkpoint(model_save_path)
        else:
            # Tokenizer
            try:
                self.tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False, cache_dir="GPT_Trainer/llama2", token=token)
                # self.tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", use_fast=False, cache_dir="GPT_Trainer/gpt-j-6B")
            except OSError:
                raise FileNotFoundError("Token not found in .env file or user does not have access to Llama 2 weights with that token. Please add your Hugging Face token to the .env file.")
            
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.pad_token = torch.tensor([self.tokenizer.pad_token_id])
            
            # Set max sequence length
            self.tokenizer.model_max_length = model_max_length
            
            # Get model
            if model_size == "small":
                self.model = transformers.LlamaForCausalLM(config=transformers.LlamaConfig.from_dict({
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
                    "vocab_size": self.tokenizer.vocab_size,
                    "attention_type": attention_type,
                }))
            elif model_size == "large":
                self.model = transformers.LlamaForCausalLM(config=transformers.LlamaConfig.from_dict({
                    "_name_or_path": "meta-llama/Llama-2-7b-hf",
                    "architectures": [
                        "LlamaForCausalLM"
                    ],
                    "bos_token_id": 1,
                    "eos_token_id": 2,
                    "hidden_act": "silu",
                    "hidden_size": 1024*3,
                    "initializer_range": 0.02,
                    "intermediate_size": 1024*6,
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
                    "vocab_size": self.tokenizer.vocab_size,
                    "attention_type": attention_type,
                }))
            else:
                raise RuntimeError(f"Model size must be small or large, but got {model_size}")
            
            
            # Replace all self attention layers with the cosine attention layer
            for i, layer in enumerate(self.model.model.layers):
                old = layer
                self.model.model.layers[i] = LlamaDecoderLayer(self.model.config, layer_idx=i).to(layer.self_attn.q_proj.weight.device)
                self.model.model.layers[i].self_attn.layer_num = i
                del old

            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1_000_000_000
            print(f"Number of parameters: {num_params:.2f}B")
                    
                    
                    
            # Add attention type to the config
            self.attention_type = attention_type
            self.mlp_type = mlp_type
            
            
            
            
            # Put the model on the desired device
            if dev != "cpu":
                # Initialize the environment
                if not dist.is_initialized():
                    init_distributed()
                
                try:
                    local_rank = int(os.environ['LOCAL_RANK'])
                except KeyError:
                    local_rank = 0
                    print("LOCAL_RANK not found in environment variables. Defaulting to 0.")

                self.model = DDP(self.model.cuda(), device_ids=[local_rank], find_unused_parameters=False)
            else:
                self.model = self.model.cpu()
            
            
            
            # Optimizer
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=self.weight_decay, eps=1e-7)

            # LR Scheduler
            self.scheduler = get_scheduler(self.optimizer, warmup_steps=warmup_steps, total_steps=self.num_steps)

            # Step starts at 0
            self.step_ckpt = 0
            
            # Wandb id is None
            self.wandb_id = None
            
            # Base model reference for DDP
            if self.dev == "cpu":
                self.model_ref = self.model
            else:
                self.model_ref = self.model.module
            
            
        
    def prepare_data(self, batch):
        # Tokenize the batch
        batch = self.tokenizer([i["text"] for i in batch], truncation=True, padding="longest", padding_side=self.padding_side, return_tensors="pt", max_length=self.tokenizer.model_max_length)

        # # Max length of the input (+1 for the extra pad token), but not more than the model's max length
        # max_length = min(max([len(x) for x in batch["input_ids"]]), self.tokenizer.model_max_length)
        
        # Labels are the input ids shifted by 1
        batch["labels"] = batch["input_ids"].clone()[:, 1:]
        batch["input_ids"] = batch["input_ids"][:, :-1]
        batch["attention_mask"] = batch["attention_mask"][:, :-1].bool()
        
        # for i in range(len(batch)):
        #     ### Random window of the max 6length number of tokens ###
        #     if len(batch["input_ids"][i]) > max_length:
        #         start = np.random.randint(0, len(batch[i]["input_ids"]) - max_length)
        #         batch[i]["input_ids"] = batch[i]["input_ids"][start:start+max_length]
        #         batch[i]["attention_mask"] = batch[i]["attention_mask"][start:start+max_length]
            
        #     ### Add a pad token to the end without mask to make the model stop itself
        #     batch[i]["input_ids"] = torch.cat([batch[i]["input_ids"], self.pad_token])
        #     batch[i]["attention_mask"] = torch.cat([batch[i]["attention_mask"], torch.tensor([0])])
        
        #     ### Pad the input to max length
        #     batch[i]["input_ids"] = torch.cat([batch[i]["input_ids"], self.pad_token.repeat(max_length+1 - len(batch[i]["input_ids"]))])
        #     batch[i]["attention_mask"] = torch.cat([batch[i]["attention_mask"], torch.zeros(max_length+1 - len(batch[i]["attention_mask"]), dtype=torch.long)]).bool()
            
        #     ### Labels are input ids shifted by one. Remove the last token from the others to match the labels
        #     batch[i]["labels"] = batch[i]["input_ids"].clone()[1:]
        #     batch[i]["input_ids"] = batch[i]["input_ids"][:-1]
        #     batch[i]["attention_mask"] = batch[i]["attention_mask"][:-1]

        # When all the sequence lengths are the same, the mask will be all True.
        # Annoyingly, this causes a bug with the attention mask. To get around this
        # issue without changing the transformers code, I am adding a single "False"
        # to one of the positions.
        if torch.all(batch["attention_mask"] == True):
            batch["attention_mask"][0, -1] = False
                    
        # Stack the data
        return batch
        
        
        
        
    def __call__(self):
        if self.finetune_:
            self.finetune()
        else:
            self.train_model()
            
            
            
            
    def train_model(self):
        # self.train_model_("Traxap/Pile_Tokenized", self.num_steps, self.step_ckpt)
        self.train_model_(self.num_steps, self.step_ckpt)
        
        # self.train_model_("gmongaras/Pile_Llama_Tokenized", self.num_steps, self.step_ckpt)
        # self.train_model_("gmongaras/BERT_Base_Cased_512_Dataset_Mapped", self.num_steps, self.step_ckpt)
        # self.train_model_("gmongaras/dummy_text_dataset", self.num_steps, self.step_ckpt)
        
        
        
        
        
    def train_model_(self, num_steps, step_shift):
        # Cache dirs
        # cache_path = "/users/gmongaras/work/datasets/data_cache/"
        cache_path = "cache"
        os.environ["HF_HOME"] = cache_path
        os.environ["HF_HUB_CACHE"] = cache_path
        # cache_path = "BERT_Trainer/data_cache/dataset_mapped"
        # cache_path = "GPT_Trainer/data_cache/dataset_mapped"
        
        # Load in datasets
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        if self.dataset == "HuggingFaceFW/fineweb":
            name = "CC-MAIN-2024-51"
        else:
            name = None
        self.dataset_ = datasets.load_dataset(self.dataset,
                name=name,
                cache_dir=cache_path,
                num_proc=16,
                split="train",
                download_config=datasets.DownloadConfig(
                    max_retries=20,
                    cache_dir=cache_path,
                ))
        
        # Load dummy data
        # tokenized_dataset = datasets.load_from_disk("BERT_Trainer/data_cache/dummy_dataset")
        
        
        if self.dataset == "gmongaras/dummy_text_dataset":
            def tokenize_function(examples):
                return self.tokenizer(examples["text"], truncation=False)
            self.dataset_ = self.dataset_.map(
                tokenize_function,
                remove_columns=["text"],
                cache_file_name="dummy_tokenized_dataset",
            )

        # Test/train split
        if self.dataset in [
                "gmongaras/SlimPajama-627B_Reupload",
            ]:
            self.dataset_test = datasets.load_dataset(self.dataset,
                name=name,
                cache_dir=cache_path,
                num_proc=16,
                split="test",
                download_config=datasets.DownloadConfig(
                    max_retries=20,
                    cache_dir=cache_path,
                ))
        else:
            data_split = self.dataset_.train_test_split(test_size=self.test_per, seed=123)
            self.dataset_ = data_split["train"]
            self.dataset_test = data_split["test"]
        
        # Convert data to torch
        self.dataset_.set_format(type="torch", columns=["text"])
        self.dataset_test.set_format(type="torch", columns=["text"])
        
        # PyTorch random sampler
        random_sampler = torch.utils.data.RandomSampler(self.dataset_, replacement=True, num_samples=(num_steps-step_shift)*self.batch_size)
        
        # PyTorch data loader
        data_loader = torch.utils.data.DataLoader(
            self.dataset_, 
            sampler=random_sampler,
            batch_size=self.batch_size, 
            collate_fn=lambda x: x,
            
            num_workers=10,
            prefetch_factor=10,
            persistent_workers=True,
        )
        
        
        # Train mode
        self.model.train()
        
        # Initialize wandb run
        if is_main_process():
            wandb.init(
                project="Gated_Attention",
                name=self.wandb_name,
                notes=None, # May add notes later
                
                # Resume training if checkpoint exists
                resume="must" if self.wandb_id is not None else None,
                id=self.wandb_id,
            )
            # wandb.watch(self.model, log_freq=self.log_steps)
            
            # Save wandb run id
            self.wandb_id = wandb.run.id
        
        # Automatic mixed precision
        if self.use_amp:
            grad_scaler = torch.amp.GradScaler("cuda")
    
        
        batch_loss = 0
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Training loop
        step = step_shift
        for step, batch in enumerate(tqdm(data_loader, initial=step_shift, total=num_steps)) if is_main_process() else enumerate(data_loader):
            step += step_shift
                
            # Augment input
            batch = self.prepare_data(batch)
                
            # Get input and labels
            input_ids = batch["input_ids"].to(self.model.device)
            attention_mask = batch["attention_mask"].to(self.model.device)
            labels = batch["labels"].to(self.model.device)
        
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16) if self.use_amp else nullcontext():
                outputs = self.model(input_ids, attention_mask=attention_mask).logits
                
                # Mask labels with -100 where the attention mask is 0. Note that the mask needs to be shifted by one to match the labels
                labels = torch.where(attention_mask, labels, torch.tensor(-100).to(labels.device))
                
                # Loss
                loss = loss_fct(outputs.view(-1, self.model_ref.config.vocab_size), labels.view(-1).to(outputs.device))
                
                # # Perplexity calculation
                # # https://huggingface.co/docs/transformers/en/perplexity
                # with torch.no_grad():
                #     # loss is calculated using CrossEntropyLoss which averages over valid labels
                #     # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
                #     # to the left by 1.
                #     neg_log_likelihood = torch.nn.functional.nll_loss(
                #         outputs.view(-1, self.model_ref.config.vocab_size), 
                #         labels.view(-1).to(outputs.device)
                #     )
                #     # Accumulate the total negative log-likelihood and the total number of tokens
                #     num_valid_tokens = (labels != -100).sum().item()  # number of valid tokens in target_ids
                #     batch_size = labels.shape[0]
                #     num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
                #     nll_sum = neg_log_likelihood * num_loss_tokens
                #     n_tokens = num_loss_tokens
                #     avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
                #     ppl = torch.exp(avg_nll).cpu().detach().item()

            # Backpropagate loss
            if self.use_amp:
                grad_scaler.scale(loss).backward()
            else:
                loss.backward()
                
            # Clip gradients
            if self.use_amp:
                grad_scaler.unscale_(self.optimizer)
            if self.clipping_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping_value)
            
            # Take optimizer step
            if self.use_amp:
                grad_scaler.step(self.optimizer)
            else:
                self.optimizer.step()
            
            # Update scheduler
            self.scheduler.step(step)
            
            # Step the gradient scaler
            if self.use_amp:
                grad_scaler.update()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            
            
            # Update batch loss
            batch_loss += loss.item()/self.log_steps
            
            
            
            
            # Log wandb
            if (step) % self.log_steps == 0:
                if is_main_process():                    
                    wandb.log({
                        "loss": batch_loss,
                        "perplexity": torch.exp(torch.tensor(batch_loss)),
                        "lr": self.optimizer.param_groups[0]['lr'],
                    },
                    step=step)
                
                batch_loss = 0
                
            
            # Break if we have reached the max number of steps
            if (step) >= self.num_steps:
                break
            
            
            
            
            if step % self.num_save_steps == 0:
                self.save_model(step)
                
                
                
            # Clear cache
            # torch.cuda.empty_cache()

            if step == self.num_steps_early_stop:
                break





            # Testing the model
            if step % self.num_steps_test == 0 and step > 10:
                with torch.no_grad():
                    # Put model in eval mode
                    self.model.eval()

                    # Create sampler and datalaoder
                    test_data_loader = torch.utils.data.DataLoader(
                        self.dataset_test,
                        shuffle=True,
                        batch_size=self.batch_size, 
                        collate_fn=lambda x: x,
                        
                        num_workers=10,
                        prefetch_factor=10,
                        persistent_workers=True,
                    )

                    # Average loss and ppl
                    avg_test_loss = 0
                    avg_test_ppl = 0
                    total_batches = 0

                    # Testing loop
                    for num, batch in enumerate(tqdm(test_data_loader)):
                        # Augment input
                        batch = self.prepare_data(batch)
                            
                        # Get input and labels
                        input_ids = batch["input_ids"].to(self.model.device)
                        attention_mask = batch["attention_mask"].to(self.model.device)
                        labels = batch["labels"].to(self.model.device)
                    
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16) if self.use_amp else nullcontext():
                            outputs = self.model(input_ids, attention_mask=attention_mask).logits
                            
                            # Mask labels with -100 where the attention mask is 0. Note that the mask needs to be shifted by one to match the labels
                            labels = torch.where(attention_mask, labels, torch.tensor(-100).to(labels.device))
                            
                            # Loss
                            loss = loss_fct(outputs.view(-1, self.model_ref.config.vocab_size), labels.view(-1).to(outputs.device))

                        # Perplexity
                        ppl = loss.exp().item()
                        loss = loss.item()

                        # Accumulate loss and ppl
                        avg_test_loss += loss
                        avg_test_ppl += ppl
                        total_batches += 1

                    # Get the averages
                    avg_test_loss /= total_batches
                    avg_test_ppl /= total_batches

                    # Log to wandb 
                    if is_main_process():
                        wandb.log({
                            "test_loss": avg_test_loss,
                            "test_perplexity": avg_test_ppl,
                        },
                        step=step)

                    del test_data_loader

                    # Put model in train mode
                    self.model.train()

                
            
                
                
    def save_model(self, step):
        if is_main_process():
            # Save the model
            self.model_ref.save_pretrained(self.model_save_path)
            self.tokenizer.save_pretrained(self.model_save_path)
            
            # Save the optimizer
            torch.save(self.optimizer.state_dict(), os.path.join(self.model_save_path, "optimizer.pt"))
            
            # Save the scheduler
            torch.save(self.scheduler.state_dict(), os.path.join(self.model_save_path, "scheduler.pt"))
            
            # Save the config
            torch.save({
                "learning_rate": self.learning_rate,
                "warmup_steps": self.warmup_steps,
                "num_steps": self.num_steps,
                "wandb_name": self.wandb_name,
                "log_steps": self.log_steps,
                "use_amp": self.use_amp,
                "dev": self.dev,
                "clipping_value": self.clipping_value,
                "weight_decay": self.weight_decay,
                "attention_type": self.attention_type,
                "mlp_type": self.mlp_type,
                "step_ckpt": step,
                "wandb_id": self.wandb_id,
            }, os.path.join(self.model_save_path, "config.pt"))
            
            # Save the tokenizer
            torch.save(self.tokenizer, os.path.join(self.model_save_path, "tokenizer.pt"))
            
            
            
    def load_checkpoint(self, checkpoint_path):
        # Load the model
        self.model = transformers.LlamaForCausalLM.from_pretrained(checkpoint_path.replace(" ", "_"))
        
        # Load the config
        config = torch.load(os.path.join(checkpoint_path, "config.pt"))
        self.learning_rate = config["learning_rate"]
        self.warmup_steps = config["warmup_steps"]
        self.num_steps = config["num_steps"]
        self.wandb_name = config["wandb_name"]
        self.log_steps = config["log_steps"]
        self.use_amp = config["use_amp"]
        self.dev = config["dev"]
        self.clipping_value = config["clipping_value"]
        self.weight_decay = config["weight_decay"]
        self.step_ckpt = config["step_ckpt"]
        self.wandb_id = config["wandb_id"]
        self.num_samples = config.get("num_samples", 0)
        self.attention_type = config["attention_type"]
        self.mlp_type = config["mlp_type"]
        
        # Replace all self attention layers with the cosine attention layer
        for i, layer in enumerate(self.model.model.layers):
            old = layer
            self.model.model.layers[i] = LlamaDecoderLayer(self.model.config, layer_idx=i).to(layer.self_attn.q_proj.weight.device)
            self.model.model.layers[i].self_attn.layer_num = i
            del old

        # Load in params
        self.model.load_state_dict(safetensors.torch.load_file(self.model_save_path + "/model.safetensors"), strict=True)
        
        # Load the tokenizer
        self.tokenizer = torch.load(os.path.join(checkpoint_path, "tokenizer.pt"), weights_only=False)            
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.pad_token = torch.tensor([self.tokenizer.pad_token_id])
        # Set max sequence length
        self.tokenizer.model_max_length = self.model_max_length 
            
            
        # Put the model on the desired device
        if self.dev != "cpu":
            if self.finetune_:
                self.model = self.model.cuda()
                
                self.model_ref = self.model
            else:
                # Initialize the environment
                if not torch.distributed.is_initialized():
                    init_distributed()
                
                try:
                    local_rank = int(os.environ['LOCAL_RANK'])
                except KeyError:
                    local_rank = 0
                    print("LOCAL_RANK not found in environment variables. Defaulting to 0.")

                self.model = DDP(self.model.cuda(), device_ids=[local_rank], find_unused_parameters=False)
                self.model_ref = self.model.module
        else:
            self.model = self.model.cpu()
            
            self.model_ref = self.model
            
            
            
        # Load the optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), weight_decay=self.weight_decay, eps=1e-7)
        self.optimizer.load_state_dict(torch.load(os.path.join(checkpoint_path, "optimizer.pt"), map_location=self.model.device))
        
        # Load the scheduler
        self.scheduler = get_scheduler(self.optimizer, warmup_steps=self.warmup_steps, total_steps=self.num_steps)
        self.scheduler.load_state_dict(torch.load(os.path.join(checkpoint_path, "scheduler.pt"), map_location=self.model.device))


# GPT in my dreams UwU
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⡿⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⡟⠀⣠⣀⠙⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣄⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⡟⠀⣼⣿⣿⣿⣦⣄⠙⠻⣿⣿⣿⣿⣿⣿⣿⠀⢻⣷⣦⣈⠙⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠛⠛⠛⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⠃⢰⣿⣿⣿⣿⣿⣿⣿⣦⡍⠙⠉⣁⣠⣤⣤⣄⡀⢻⣿⣿⣿⣦⣄⣈⠙⠿⢿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠋⣀⣠⣴⣶⣿⣷⡄⠘⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⡏⠀⣼⣿⣿⣿⣿⣿⣿⣿⣿⣄⣀⠛⢿⣿⣿⣿⣿⣷⣾⣿⣿⣿⣿⣿⣿⣷⣶⣄⠛⣿⣿⣿⡿⠟⠋⣠⣴⣾⣿⣿⣿⣿⣿⣿⡇⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⡇⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⣌⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣄⠘⣿⠋⠀⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⠁⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣦⡄⠉⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣤⣦⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⠀⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠁⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⡇⠘⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢻⣿⡀⢻⣿⣿⣿⠏⢠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣷⡀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡘⣿⠃⣸⣿⣿⠏⢀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⡿⠿⠿⠛⠃⣠⣿⣿⡿⠟⠁⢀⣀⣀⡀⠉⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣶⣿⡿⠋⢀⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣷⡈⢶⣶⣿⣿⣿⣿⣦⣤⣾⣿⣿⣿⣿⣷⣀⢘⣿⣿⣿⣿⣿⣿⣿⣿⡿⠛⠉⠀⣀⣀⣀⠀⠉⠻⣿⣿⣿⣿⣿⣿⠟⠀⠀⠛⠛⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣷⣄⡛⠟⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⣠⣶⣿⣿⣿⣿⣿⣷⣄⠈⣿⣿⣿⣿⣿⣶⣾⣿⡟⠁⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⡟⢁⣾⡟⠿⠛⠉⢻⣿⣿⣿⣿⣧⣀⡀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠟⠁⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⡿⠀⣿⣿⣿⣿⣿⡁⣉⣁⣤⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠛⠛⠛⢿⡿⠿⢿⣿⣿⡀⠠⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⡟⠀⠚⠛⣉⣉⣉⡉⠛⢿⣿⣿⣿⣿⣿⣿⡿⢿⣿⠿⢿⣿⣿⡏⣿⣿⣿⣿⣿⣧⣴⣶⣧⡀⢉⣠⣶⣿⣿⣿⣷⡀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣷⣶⣿⣿⣿⣿⣿⣷⣦⡀⠙⠻⢿⣿⣿⣿⣧⣌⠉⣠⣬⣍⠋⢁⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣾⣿⠿⢿⣿⣿⣿⡇⠘⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣤⣀⣉⠙⠛⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠿⠟⠛⠋⠉⠠⠤⣤⣴⣶⣦⣤⣤⣄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⡀⠠⣤⣤⣤⣤⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠿⣿⣿⣿⠃⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⡀⠉⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠋⣉⣠⣤⣶⣶⣤⣤⣄⠀⠸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠀⣸⣿⣿⣿⣿⣿⣿⣿⣿⠛⢁⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⡈⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁⣴⣿⣿⣿⣿⣿⣿⣿⡟⢁⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡀⢹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⣀⣠⣤⡄⠸⣿⣿⣿⣿⣇⠸⣿⡏⢹⣿⣿⡿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣧⡄⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠈⢿⣿⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠀⣿⣿⣿⣿⣿⣦⣈⠁⠘⠿⣿⡇⢸⠿⠟⢉⣠⣿⣿⣿⣿⣿⣷⡀⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⣧⡄⠹⣿⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣶⣦⣤⣤⣤⡆⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇⠈⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁⠹⣿⠃⢰⣿⣷⣄⠘⣿⣿⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠀⣰⡀⠈⠀⣿⣿⣿⣿⣄⠈⢻⣿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⡏⢻⣿⣿⣿⣿⣇⠹⣿⣿⣿⣿⣿⣿⣿⡿⠁⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠁⣰⣿⣇⠀⢰⣿⣿⣿⣿⣿⣇⠈⢿⣿⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣧⠘⣿⣿⣿⣿⣿⣄⠻⣿⣿⣿⣿⡿⠟⢀⠰⠻⠿⠿⣿⣿⣿⣿⣿⣿⣿⡟⢀⣼⣿⣿⣿⢠⣿⣿⣿⣿⣿⣿⣿⣇⠈⢻⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠛⠀⣿⣿⣿⣿⣿⣿⣿⣿⡄⢹⣿⣿⣿⣿⣿⣶⣤⣤⣤⣤⣴⣾⣿⣶⡶⠂⣴⣿⣿⣿⣿⡿⠟⠉⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡆⠈⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠋⢀⣰⣶⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣀⠘⠿⢿⠿⠛⠁⣀⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇⠀⣿⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠁⣠⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠛⢁⣠⣤⣤⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⢸⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⢸⣿⣿⣿⣿⡀⢿⣿⣿⣿⣿⣿⣿⣿⣿⡀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠘⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠀⣼⣿
# ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⢼⣿⣿⣿⣿⡇⠘⣿⣿⣿⣿⣿⣿⣿⣿⡇⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⠙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠁⣴⣿⣿
