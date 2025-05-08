import torch
import datasets
import os
import transformers
try:
    from GPT_Trainer.Trainer import Trainer
except ModuleNotFoundError:
    from Trainer import Trainer




def main():
    # Create the model trainer
    batch_size=36 # Total batch size across all gpus (that is, a batch size of 128 with 2 gpus has a gpu batch size of 64)
    learning_rate=1e-4
    warmup_steps=10_000
    num_steps=1_000_000
    num_steps_early_stop=100_000
    dev="gpu"
    # wandb_name="fineweb_gated_softmax_no_gate_rmsnorm_softmax_35bs_2gpu_1024seqlen"
    # wandb_name="fineweb_gated_softmax_out_gate_35bs_2gpu_1024seqlen"
    # wandb_name="fineweb_softmax_35bs_2gpu_1024seqlen"
    wandb_name="fineweb_large_gated_softmax_no_gate_L2norm_nodivS_36bs_4gpu_1024seqlen"
    log_steps=10
    use_amp=True
    # attention_type="gated_softmax_no_gate_rmsnorm"
    attention_type="gated_softmax_no_gate_L2norm_nodivS"
    # attention_type="softmax"
    # dataset="gmongaras/EleutherAI_the_pile_deduplicated"
    # dataset="gmongaras/SlimPajama-627B_Reupload"
    dataset="HuggingFaceFW/fineweb"
    # dataset = "gmongaras/dummy_text_dataset"
    mlp_type="normal" # gelu or normal
    clipping_value=None
    weight_decay=0.01
    model_save_path = "models/fineweb_large_gated_softmax_no_gate_L2norm_nodivS_36bs_4gpu_1024seqlen"
    # model_save_path = "models/del"
    num_save_steps = 1_000
    keep_dataset_in_mem = False
    model_max_length = 1024
    test_per = 0.001
    num_steps_test = 10_000
    model_size = "large" # "small" (~300 million) or "large" (~2 billion)
    test_loss = True

    
    # Load in a checkpoint
    load_checkpoint = False
    checkpoint_path = "models/fineweb_large_gated_softmax_no_gate_L2norm_nodivS_36bs_4gpu_1024seqlen/"


    """
    # Create the model trainer
    batch_size=20 # Total batch size across all gpus (that is, a batch size of 128 with 2 gpus has a gpu batch size of 64)
    learning_rate=1e-4
    warmup_steps=10_000
    num_steps=1_000_000
    num_steps_early_stop=30_000
    dev="gpu"
    "fineweb_noAproj_dodt_noDgate_noznorm_noinconv_normsnorm_1expand_plusplus_20bs_1gpu_1024seqlen"
    wandb_name="fineweb_softmax_20bs_1gpu_1024seqlen"
    log_steps=10
    use_amp=True
    attention_type="softmax"
    # dataset="gmongaras/EleutherAI_the_pile_deduplicated"
    # dataset="gmongaras/SlimPajama-627B_Reupload"
    dataset="HuggingFaceFW/fineweb"
    # dataset = "gmongaras/dummy_text_dataset"
    mlp_type="normal" # gelu or normal
    clipping_value=None
    weight_decay=0.01
    model_save_path = "models/fineweb_softmax_20bs_1gpu_1024seqlen"
    # model_save_path = "models/del"
    num_save_steps = 1_000
    keep_dataset_in_mem = False
    model_max_length = 1024
    test_per = 0.001
    num_steps_test = 10_000
    model_size = "small" # "small" (~300 million) or "large" (~2 billion)
    test_loss = False
    """


    
    trainer = Trainer(
        dataset=dataset,
        model_size=model_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_steps=num_steps,
        num_steps_early_stop=num_steps_early_stop,
        dev=dev,
        wandb_name=wandb_name,
        log_steps=log_steps,
        use_amp=use_amp,
        attention_type=attention_type,
        mlp_type=mlp_type,
        clipping_value=clipping_value,
        weight_decay=weight_decay,
        model_save_path=model_save_path,
        num_save_steps=num_save_steps,
        keep_dataset_in_mem=keep_dataset_in_mem,
        load_checkpoint=load_checkpoint,
        checkpoint_path=checkpoint_path,
        model_max_length=model_max_length,
        test_per=test_per,
        num_steps_test=num_steps_test,
        test_loss=test_loss
    )
    
    # Train model
    trainer()





if __name__ == "__main__":
    main()
