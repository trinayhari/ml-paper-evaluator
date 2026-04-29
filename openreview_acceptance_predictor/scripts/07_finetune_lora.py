import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.io_utils import read_jsonl
from src.text import truncate_words


def format_example(r):
    label = 'ACCEPT' if int(r['label']) == 1 else 'REJECT'
    text = truncate_words(r.get('title','') + '\n' + r.get('abstract','') + '\n' + r.get('paper_text',''), 2500)
    return {'text': f"Predict whether this ML paper will be accepted. Return ACCEPT or REJECT and a short rationale.\n\nPaper:\n{text}\n\nAnswer: {label}"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train', required=True); p.add_argument('--dev', required=True); p.add_argument('--model', required=True); p.add_argument('--out', required=True)
    args = p.parse_args()
    try:
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError('LoRA fine-tuning requires both peft and trl to be installed') from exc
    train = Dataset.from_list([format_example(r) for r in read_jsonl(args.train)])
    dev = Dataset.from_list([format_example(r) for r in read_jsonl(args.dev)])
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_kwargs = {}
    if torch.cuda.is_available():
        model_kwargs['device_map'] = 'auto'
        model_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    peft = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM', target_modules=['q_proj','v_proj'])
    tr_args = SFTConfig(
        output_dir=args.out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=20,
        eval_strategy='steps',
        eval_steps=200,
        save_steps=200,
        report_to='none',
        max_length=2048,
        fp16=torch.cuda.is_available(),
        bf16=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=tr_args,
        train_dataset=train,
        eval_dataset=dev,
        processing_class=tok,
        peft_config=peft,
    )
    trainer.train()
    trainer.save_model(args.out)

if __name__ == '__main__':
    main()
