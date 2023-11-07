import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer

def train():
    # Check for available GPU devices and list them
    if torch.cuda.is_available():
        available_gpus = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        print("Available CUDA devices:", available_gpus)
    else:
        available_gpus = ['cpu']
        print("No CUDA devices available. Using CPU.")
    
    # Load dataset
    train_dataset = load_dataset("tatsu-lab/alpaca", split="train")
    
    # Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained("daryl149/llama-2-7b-chat-hf", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model for quantization and load pretrained weights
    quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
    model = AutoModelForCausalLM.from_pretrained(
        "daryl149/llama-2-7b-chat-hf",
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map={available_gpus[0]: None} if available_gpus[0].startswith('cuda') else 'cpu',
        quantization_config=quantization_config
    )
    
    # Resize token embeddings in case the tokenizer's vocabulary has changed
    model.resize_token_embeddings(len(tokenizer))
    
    # Prepare model for k-bit training (replacement for deprecated int8 training)
    model = prepare_model_for_kbit_training(model)
    
    # Define PEFT configuration
    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)
    
    # Define training arguments
    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir="llama-finetuned-7b2",
        per_device_train_batch_size=2,  # Reduced batch size to manage memory
        gradient_accumulation_steps=16,  # Add gradient accumulation
        optim="adamw_torch",
        logging_steps=100,
        learning_rate=2e-4,
        fp16=use_fp16,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        num_train_epochs=1,
        save_strategy="epoch",
        push_to_hub=True,
    )
    
    # Initialize the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
        peft_config=peft_config,
    )
    
    # Start training
    trainer.train()
    
    # Push model to the hub
    trainer.push_to_hub()

if __name__ == "__main__":
    train()
