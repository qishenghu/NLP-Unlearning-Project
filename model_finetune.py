from datasets import load_dataset, concatenate_datasets
from transformers import GPT2Tokenizer, GPT2LMHeadModel, T5Tokenizer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from tqdm import tqdm
import argparse
import os

# GPT models to be used
# GPT2-small(124M): openai-community/gpt2
# GPT2-medium(355M): openai-community/gpt2-medium
# GPT2-large(774M): openai-community/gpt2-large
# GPT2-xl(1.5B): openai-community/gpt2-xl

# Encoder-Decoder models to be used
# T5-small (60M): google-t5/t5-small
# T5-base (220M): google-t5/t5-base
# T5-large (770M): google-t5/t5-large
# T5-xl (3B): google/t5-v1_1-xl

def model_name_to_id(model_name):
    name2id = {
        "gpt2-small": "openai-community/gpt2",
        "gpt2-medium": "openai-community/gpt2-medium",
        "gpt2-large": "openai-community/gpt2-large",
        "gpt2-xl": "openai-community/gpt2-xl",
        "t5-small": "google-t5/t5-small",
        "t5-base": "google-t5/t5-base",
        "t5-large": "google-t5/t5-large",
        "t5-xl": "google/t5-v1_1-xl"
    }
    return name2id[model_name]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2-large")
    parser.add_argument("--output_dir", type=str, default="./target_model/")
    parser.add_argument("--corpus", type=str, default="news", choices=["books", "news"])
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--use_wandb", type=bool, default=False)
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    
    
    # 1. Load datasets
    if args.corpus == "news":
        forget_dataset = load_dataset("muse-bench/MUSE-News", "raw", split="forget")
        retain_dataset = load_dataset("muse-bench/MUSE-News", "raw", split="retain2")
    elif args.corpus == "books":
        forget_dataset = load_dataset("muse-bench/MUSE-Books", "raw", split="forget")
        retain_dataset = load_dataset("muse-bench/MUSE-Books", "raw", split="retain2")
    full_dataset = concatenate_datasets([forget_dataset, retain_dataset])

    # 2. Load tokenizer and model
    model_id = model_name_to_id(args.model_name)
    if "gpt" in model_id.lower():
        tokenizer = GPT2Tokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_id)
    elif "t5" in model_id.lower():
        tokenizer = T5Tokenizer.from_pretrained(model_id)
        model = T5ForConditionalGeneration.from_pretrained(model_id)




    # 3. Tokenize
    # Roughly 73%+ exceed 512 tokens, so i use chunking

    # T5 models require task-specific prefixes to be added to the input text. 
    # Since your task involves language modeling similar to GPT-2, you can use a generic prefix like "lm: " 
    # to indicate a language modeling task.​
    def tokenize_function(examples, use_t5=False):
        input_ids_list = []
        labels_list = []
        for text in examples["text"]:
            if use_t5:
                text = "lm: " + text  # Add task prefix for T5
            tokens = tokenizer(text, truncation=False)["input_ids"]

            # Add task prefix to each chunk for T5
            for i in range(0, len(tokens), 512):
                chunk = tokens[i:i + 512]
                if use_t5:
                    prefix_tokens = tokenizer("lm: ", add_special_tokens=False)["input_ids"]
                    chunk = prefix_tokens + chunk
                    if len(chunk) > 512:
                        chunk = chunk[:512]
                if len(chunk) < 512:
                    chunk += [tokenizer.pad_token_id] * (512 - len(chunk))
                input_ids_list.append(chunk)
                
                # Create labels with padding tokens set to -100
                labels = chunk.copy()
                labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]
                labels_list.append(labels)
        tokenized = {"input_ids": input_ids_list, "labels": labels_list} if use_t5 else {"input_ids": input_ids_list}
        return tokenized

    use_t5 = "t5" in model_id.lower()
    tokenized_dataset = full_dataset.map(
                                        lambda examples: tokenize_function(examples, use_t5=use_t5), 
                                        batched=True, 
                                        remove_columns=full_dataset.column_names
                                        )

    

    # 4. Define data collator
    if use_t5:
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,   # where to save model
        evaluation_strategy="no",          # no evaluation during training
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,               # or early stop manually
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        report_to="wandb" if args.use_wandb else None,
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 7. Train
    trainer.train()

    # Save final model
    trainer.save_model(output_dir)

    # Save tokenizer
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()