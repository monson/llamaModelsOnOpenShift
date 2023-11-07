import logging
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    # Check for available GPU devices and list them
    if torch.cuda.is_available():
        available_gpus = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        logger.info(f"Available CUDA devices: {available_gpus}")
    else:
        logger.warning("No CUDA devices available. Using CPU.")
        available_gpus = ['cpu']
    
    # Load dataset
    train_dataset = load_dataset("tatsu-lab/alpaca", split="train")
    logger.info("Dataset loaded successfully.")
    
    # Prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained("daryl149/llama-2-7b-chat-hf", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer prepared successfully.")
    
    # Prepare model for quantization and load pretrained weights
    quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
    model = AutoModelForCausalLM.from_pretrained(
        "daryl149/llama-2-7b-chat-hf",
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map={available_gpus[0]: None} if available_gpus[0].startswith('cuda') else 'cpu',
        quantization_config=quantization_config
    )
    model.resize_token_embeddings(len(tokenizer))
    logger.info("Model loaded and token embeddings resized successfully.")
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Define PEFT configuration
    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)
    logger.info("PEFT configuration prepared successfully.")
    
    # Define training arguments
    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir="llama-finetuned-7b2",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
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
    logger.info(f"Training arguments: {training_args}")
    
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
    logger.info("Trainer initialized successfully.")
    
    # Start training
    trainer.train()
    logger.info("Training started.")
    
    # Push model to the hub (optional)
    trainer.push_to_hub()
    logger.info("Model pushed to Hugging Face Hub.")

if __name__ == "__main__":
    train()
