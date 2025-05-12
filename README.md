# On the Expressiveness of Softmax Attention: A Recurrent Neural Network Perspective

This repo is code used for experiments in the paper "On the Expressiveness of Softmax Attention: A Recurrent Neural Network Perspective".




# Setup

This repo was trained with python 3.10. Other versions may or may not work.

To setup, first ensure you have cuda properly setup. This can be checked by running `nvidia-smi` and `nvcc -V`.

Create a virtual environment with 
```
python -m venv GatedAttnEnv
source GatedAttnEnv/bin/activate
```

Install the requirements
```
pip install -r requirements.txt
```

Then, install the version of torch for your system at `https://pytorch.org/get-started/locally/`. This repo was run on torch `2.6.0` with cuda `11.8`. Your system will likely need to use a different version of cuda. The following command install torch `2.6.0` for cuda `11.8`:
```
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```





# Running the script

The script can be run with the following:

```
torchrun --nproc_per_node=1 --master-port $PORT GPT_Trainer/train.py
```

where $PORT is just an arbitrary open port number.




# Important Scripts

Almost everything can be controlled from the `train.py` script. The following parameters are adjustable:
- `batch_size` - Global batch size across all GPUs. A batch size of 36 across 2 GPUs would mean a batch size of 18 on each GPU.
- `learning_rate` - Model learning rate
- `warmup_steps` - Number of steps to linearly increase the learning rate from 0 to `learning_rate`. After the number of steps reaches `warmup_steps`, the learning rate is linearly decreased to 0 and will hit 0 once `num_steps` update steps has been reached.
- `num_steps` - Total number of steps the model will be trained for.
- `num_steps_early_stop` - Stops the model early at this many steps. The learning rate scheduler is not changed.
- `dev` - Keep this at `gpu`
- `wandb_name` - Wandb will log the run under this name. Note that the project is defaulted to `Gated_Attention`
- `log_steps` - Wandb will log the model loss every `log_steps` number of steps
- `use_amp` - True to use AMP with bfloat16, False to stay in float32
- `attention_type` - Type of attention this model will use. All types can be found in the `LlamaDecoderLayer.py` script. The useful options are mentioned below.
- `dataset` - Name of the dataset to load in. Can be one of `gmongaras/EleutherAI_the_pile_deduplicated`, `gmongaras/SlimPajama-627B_Reupload`, `HuggingFaceFW/fineweb`.
- `mlp_type` - THe MLP type to use. We stick with `normal` which uses the normal, gated MLP block in llama. `gelu` swaps this with a GELU MLP without a gate.
- `clipping_value` - `None` for no clipping, a float value to perform gradient clipping with said value.
- `weight_decay` - Normal optimizer weight decay param
- `model_save_path` - Local path to save model checkpoints to
- `num_save_steps` - Every `num_save_steps`, the current model will be saved along with the states of the optimizer and scheduler.
- `keep_dataset_in_mem` - Keep this `False`
- `model_max_length` - The max number of tokens to train the model with.
- `test_per` - Percentage of data to make test data.
- `num_steps_test` - Every `num_steps_test`, the model will stop trained, will iterate over all test data, and calculate metrics logged to wandb.
- `model_size` - `small` for the small model (~300 million params), `large` for the large model (~2 billion params).
- `test_loss` - `True` to test the model on test data, `False` to skip this and just train the model.

The base model is llama 2. We just swap out the blocks with custom blocks. The code for these blocks can be found in the `LlamaDecoderLayer.py` script. Additionally, this script has all the `attention_type` options.

The training actually happens in `Trainer.py`.

To test models, `infer.py` can perform inference using pretrained models. However the script is very unoptimized and is only used for testing purposes.



# Experiment Info

Unless otherwise mentioned, the below are the parameters we used in our models. As our base model is llama 2, RoPE is used on the attention matrix and the MLPs follow SwiGLU.

- batch size - 36
- learning rate - 1e-4
- warmup steps - 10,000
- warmup type - linear warmup from 0, linear decay
- num steps - 1,000,000
- num steps early stop - 100,000
- AMP - enabled
- Weight decay - 0.01
- Max sequence length - 1024 for general experimetns, 4096 for the 4096 length experiment
- Test percentage - 0.001
- Optimizer - AdamW
- Adam betas - 0.9 and 0.999
- Hidden size - 1024 (3072 for the large model)
- MLP intermediate size - 2048 (6144 for the large model)
- Num attention heads - 16
- Num hidden layers - 20
- Tokenizer - llama2-7b-hf
- Gradient clipping - 1.0 clipping for gated models, no clipping for all other experiments
\end{enumerate}

Each model was trained for a maximum of 2 days. Most experiments, we use distributed data parallel to train on two 80 GB, A100 GPUs with the exception of the large model, trained on 4 GPUs, and 4096 sequence length, trained on 6 GPUs.

Note that `output gate` refers to a gate on the attention output or along the queries/columns of the attention matrix (both are equivalent mathematically). An `input gate` refers to a gate on the values or along the rows of the attention matrix (both are equivalent mathematically).

Most expierments are controlled by the `attention_type` parameter. All experimented Below are descriptions of the useful options for this parameter:
- `softmax` - Normal SDPA using basic softmax.
- `softmax_clamp_denom` - Softmax, but clamp the denominator to be >= 1
- `softmax_detach_denom` - Softmax, but detach the denominator.
- `softmax_detach_denom_gate` - Softmax, but detach the denominator and add a learnable output gate.
- `softmax_divs` - Softmax, but replace the denominator by dividing by the sequence length.
- `softmax_gate` - Softmax, but replace the denominator with an output gate
- `softmax_divS_gate` - Combines both `softmax_divs` and `softmax_gate`. Softmax without a denominator, repalced with an output gate and dividing by the sequence length.
- `softmax_divS_gatev2` - Same as `softmax_divS_gate`, but the output gate is applied on the attention matrix rather than the attention output. Should be mroe numerically stable.
- `softmax_divS_norm` - Softmax, but replace the denominator with a RMSNorm.
- `softmax_taylor_80terms` - Softmax decomposed as a taylor series of 80 terms
- `gated_softmax` - Softmax, no denominator, both an input and output gate, divide by the sequence length, and an output LayerNorm.
- `gated_softmax_no_norm` - Same as `gated_softmax`, but no norm. So only gates and divde by the sequence length.
- `gated_softmax_no_in_gate` - Same as `gated_softmax`, but no input gate. So an output gate, a norm and divde by the sequence length.
- `gated_softmax_no_out_gate` - Same as `gated_softmax`, but no output gate. So an input gate, a norm and divde by the sequence length.
- `gated_softmax_no_out_gate_no_norm` - Same as `gated_softmax`, but no output gate and no norm. So an input gate and divde by the sequence length.
- `gated_softmax_no_in_gate_no_norm` - Same as `gated_softmax`, but no input gate and no norm. So an output gate and divde by the sequence length. This is the same as `softmax_gate`.
- `gated_relu_no_in_gate_no_norm` - Similar to `gated_softmax`, but no input gate and no norm and ReLU linear attention is used. So the exponential is replaced by a ReLU kernel, an output gate is used and divde by the sequence length.
- `gated_softmax_no_gate` - Same as `gated_softmax` but without an input or output gate. So just LayerNorm and divide by the sequence length.
- `gated_softmax_no_gate_rmsnorm` - Same as `gated_softmax_no_gate` but with RMSNorm instead of LayerNorm.
- `gated_softmax_no_gate_L2norm_nodivS` - Same as `gated_softmax_no_gate` but with L2Norm instead of LayerNorm and no division by S.
- `gated_softmax_no_gate_L2norm_nodivS_noclamp` - Same as `gated_softmax_no_gate_L2norm_nodivS` but the inner product is not clamped. Essentially, this is just Norm(exp(QK)V)
- `gated_ReLU_no_gate_L2norm_nodivS_noclamp` - Same as `gated_softmax_no_gate_L2norm_nodivS_noclamp` but replace the inner product with a ReLU kernel. This is essentially Norm(relu(Q)relu(K)V)
- `gated_softmax_out_gate_L2norm_nodivS_noclamp` - Same as `gated_softmax_no_gate_L2norm_nodivS_noclamp`, but an output gate is added.
- `gated_softmax_post_out_gate_L2norm_nodivS_noclamp` - Same as `gated_softmax_no_gate_L2norm_nodivS_noclamp` but an output gate is added after the norm.
- `gated_softmax_no_gate_rmsnorm_nodivS` - Same as `gated_softmax_no_gate_rmsnorm`, but no division by the sequence length.
- `gated_softmax_no_gate_no_norm` - Softmax without a gate or a norm, just divison by the sequence length. This is essentially (exp(QK)V)/S.
- `linear_elu` - Linear attention with elu(x) + 1 as the activation function
- `linear_relu` - Linear attention with relu(x) as the activation function
- `linear_cosine` - Linear attention with L2Norm(X) as the activation function


Note that most of these expierments clamped the inner product to be no greater than 5, before the exponential. If a value is too large, it will be clamped to prevent numerical instability. We notice this doesn't hurt performance and helps to stabalize experiments without a norm.




# Datasets

The fineweb dataset was used for most expierments. We specifically use the "CC-MAIN-2024-51" version of this dataset. The Pile and SlimPajama were also used and have been reuploaded to utilize faster loading speeds of the more recent huggingfae library.
- [fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb/viewer/CC-MAIN-2024-51)
- [The Pile (Reuploaded)](https://huggingface.co/datasets/gmongaras/EleutherAI_the_pile_deduplicated)
- [The Pile (Original)](https://huggingface.co/datasets/EleutherAI/pile)
- [SlimPajama (Reuplaoded)](https://huggingface.co/datasets/gmongaras/SlimPajama-627B_Reupload)
- [SlimPajama (Original)](https://huggingface.co/datasets/cerebras/SlimPajama-627B)