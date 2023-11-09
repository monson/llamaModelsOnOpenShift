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
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        "daryl149/llama-2-7b-chat-hf",
        device_map = {
            "transformer.word_embeddings": "cuda:0", 
            "transformer.word_embeddings_layernorm": "cuda:0",
            "lm_head": "cuda:0",
            "transformer.h": "cuda:0",
            "transformer.ln_f": "cuda:0",
            "model.embed_tokens": "cuda:0",
            "model.layers":"cuda:0",
            "model.norm":"cuda:0"
        },
        quantization_config=quantization_config,
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
        logging_steps=10,
        learning_rate=2e-4,
        fp16=use_fp16,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        num_train_epochs=1,
        save_strategy="steps",
        save_total_limit=3,
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
    logger.info("Training started.")
    trainer.train()
    logger.info("Training completed.")
    
    # The model and training progress will be automatically saved during training at the specified intervals.
    # Save the final model and tokenizer locally after training
    trainer.save_model("llama-finetuned-7b2_final_checkpoint")
    tokenizer.save_pretrained("llama-finetuned-7b2_final_checkpoint")
    logger.info("Final model and tokenizer saved locally.")

    # Push model to the hub (optional)
    trainer.push_to_hub()
    logger.info("Model pushed to Hugging Face Hub.")

if __name__ == "__main__":
    train()
