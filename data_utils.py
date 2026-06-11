import os
import glob
import json
import numpy as np
import torch
from torch.utils.data import IterableDataset
import tiktoken
from typing import List, Union, Generator, Tuple

class TokenStreamer:
    """
    Reads text files line-by-line, tokenizes them on-the-fly, and
    yields blocks of shape (block_size + 1). The caller splits this into
    input (x) and target (y) of length block_size.
    Discards raw text immediately after tokenization to minimize memory footprint.
    """
    def __init__(self, file_paths: List[str], encoding_name: str, block_size: int, split: str = "train", split_ratio: float = 0.9):
        self.file_paths = file_paths
        self.block_size = block_size
        self.split = split
        self.split_ratio = split_ratio
        
        # Load subword tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            # Fallback/warning if offline or tiktoken fails
            print(f"Warning: Failed to load tiktoken encoding '{encoding_name}': {e}. Using gpt2 fallback.")
            self.tokenizer = tiktoken.get_encoding("gpt2")

    def __iter__(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        # Multi-worker DataLoader support
        worker_info = torch.utils.data.get_worker_info()
        file_paths = self.file_paths
        
        if worker_info is not None:
            # Distribute files among workers
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            file_paths = [path for i, path in enumerate(file_paths) if i % num_workers == worker_id]

        token_buffer = []
        block_count = 0

        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    tokens = self.tokenizer.encode(line, allowed_special="all")
                    token_buffer.extend(tokens)

                    # Yield blocks of block_size + 1 on-the-fly
                    while len(token_buffer) >= self.block_size + 1:
                        chunk = token_buffer[:self.block_size + 1]
                        token_buffer = token_buffer[self.block_size:]  # Slide context window

                        is_val_block = (block_count % 10 == 9) or (block_count < 10 and block_count % 2 == 1)
                        block_count += 1

                        if (self.split == "val" and is_val_block) or (self.split == "train" and not is_val_block):
                            x = np.array(chunk[:-1], dtype=np.int64)
                            y = np.array(chunk[1:], dtype=np.int64)
                            yield x, y

class HFStreamingStreamer:
    """
    Streams from a Hugging Face dataset, tokenizes on the fly,
    and yields input/target blocks of length block_size.
    """
    def __init__(self, dataset_name: str, split: str, encoding_name: str, block_size: int, text_column: str = "text"):
        from datasets import load_dataset, get_dataset_split_names
        self.split = split
        try:
            available_splits = get_dataset_split_names(dataset_name)
        except Exception:
            available_splits = []
        
        self.has_val_split = any(s in ["validation", "val"] for s in available_splits)
        
        # Determine actual split to load
        actual_split = split
        if split in ["val", "validation"] and not self.has_val_split:
            actual_split = "train"
            
        try:
            self.dataset = load_dataset(dataset_name, split=actual_split, streaming=True)
        except Exception as e:
            if split in ["val", "validation"]:
                alt_split = "validation" if split == "val" else "val"
                try:
                    self.dataset = load_dataset(dataset_name, split=alt_split, streaming=True)
                    self.has_val_split = True
                except Exception:
                    try:
                        self.dataset = load_dataset(dataset_name, split="train", streaming=True)
                        self.has_val_split = False
                    except Exception:
                        raise e
            else:
                raise e
        self.block_size = block_size
        self.text_column = text_column
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def __iter__(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        token_buffer = []
        record_idx = 0
        for sample in self.dataset:
            # If dataset has no validation split, partition 'train' split 90/10 manually
            if not self.has_val_split:
                is_val_sample = (record_idx % 10 == 9)
                record_idx += 1
                if self.split in ["val", "validation"] and not is_val_sample:
                    continue
                if self.split == "train" and is_val_sample:
                    continue
            else:
                record_idx += 1

            text = sample[self.text_column]
            tokens = self.tokenizer.encode(text, allowed_special="all")
            token_buffer.extend(tokens)

            while len(token_buffer) >= self.block_size + 1:
                chunk = token_buffer[:self.block_size + 1]
                token_buffer = token_buffer[self.block_size:]
                
                x = np.array(chunk[:-1], dtype=np.int64)
                y = np.array(chunk[1:], dtype=np.int64)
                yield x, y

        if len(token_buffer) > 0:
            eos_token_id = self.tokenizer.eot_token
            padding_len = (self.block_size + 1) - len(token_buffer)
            if padding_len > 0:
                token_buffer.extend([eos_token_id] * padding_len)
            chunk = token_buffer[:self.block_size + 1]
            x = np.array(chunk[:-1], dtype=np.int64)
            y = np.array(chunk[1:], dtype=np.int64)
            yield x, y

class StreamingTokenDataset(IterableDataset):
    """
    PyTorch IterableDataset for out-of-core streaming and incremental tokenization.
    """
    def __init__(self, source: Union[str, List[str]], tokenizer_name: str = "gpt2", block_size: int = 256, 
                 split: str = "train", is_hf: bool = False, text_column: str = "text"):
        super().__init__()
        self.source = source
        self.tokenizer_name = tokenizer_name
        self.block_size = block_size
        self.split = split
        self.is_hf = is_hf
        self.text_column = text_column

    def __iter__(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        if self.is_hf:
            # For HF datasets, split handling is done directly by requesting the relevant split
            return iter(HFStreamingStreamer(self.source, self.split, self.tokenizer_name, self.block_size, self.text_column))
        
        # Local files mode
        if isinstance(self.source, str):
            file_paths = glob.glob(self.source)
            file_paths.sort()
        elif isinstance(self.source, list):
            file_paths = self.source
        else:
            raise ValueError("source must be a glob pattern string or list of file paths")

        if not file_paths:
            raise FileNotFoundError(f"No files found matching the source source: {self.source}")

        # Split files list if multiple files exist
        if len(file_paths) > 1:
            split_idx = int(len(file_paths) * 0.9)
            if self.split == "train":
                active_files = file_paths[:split_idx]
            else:
                active_files = file_paths[split_idx:]
            # If split got empty files list, fallback to using all files and doing block-based split
            if not active_files:
                active_files = file_paths
        else:
            active_files = file_paths

        return iter(TokenStreamer(active_files, self.tokenizer_name, self.block_size, self.split))


def format_principles(principles_list):
    if not principles_list:
        return "- Be helpful\n- Be honest\n- Be concise"
    if isinstance(principles_list, str):
        return f"- {principles_list}"
    return "\n".join(f"- {p}" for p in principles_list)


def format_constitutional_record(record: dict, mode: str) -> str:
    # 1. Normalize record fields based on common SFT dataset schemas
    prompt = ""
    final_answer = ""
    
    # Check if Dolly-style
    if "instruction" in record and "response" in record:
        instruction = record.get("instruction", "")
        context = record.get("context", "")
        prompt = f"{instruction}\nContext: {context}" if context else instruction
        final_answer = record.get("response", "")
        
    # Check if Alpaca-style
    elif "instruction" in record and "output" in record:
        instruction = record.get("instruction", "")
        inp = record.get("input", "")
        prompt = f"{instruction}\nInput: {inp}" if inp else instruction
        final_answer = record.get("output", "")
        
    # Check if Anthropic HH-RLHF style (has 'chosen' or 'rejected')
    elif "chosen" in record:
        chosen_text = record.get("chosen", "")
        # Parse last human prompt and assistant response
        parts = chosen_text.split("\n\nHuman:")
        if len(parts) > 1:
            last_turn = parts[-1]
            if "\n\nAssistant:" in last_turn:
                prompt_part, response_part = last_turn.split("\n\nAssistant:", 1)
                prompt = prompt_part.strip()
                final_answer = response_part.strip()
                
    # Otherwise fallback to standard constitutional JSONL schema
    else:
        prompt = record.get("prompt", "")
        final_answer = record.get("final_answer", "")
        
    # Clean up outputs
    prompt = str(prompt).strip()
    final_answer = str(final_answer).strip()
    
    # Read other fields if they exist
    draft = record.get("draft", "")
    critique = record.get("critique", "")
    revision = record.get("revision", "")
    short_reasoning = record.get("short_reasoning", "")
    principles_list = record.get("principles_used", [])
    
    draft = str(draft).strip()
    critique = str(critique).strip()
    revision = str(revision).strip()
    short_reasoning = str(short_reasoning).strip()
    
    # Cross-fill revision and final_answer
    if not revision and final_answer:
        revision = final_answer
    if not final_answer and revision:
        final_answer = revision
        
    if mode == "final_only":
        return f"### Prompt:\n{prompt}\n\n### Response:\n{final_answer}"
        
    elif mode == "plan_answer":
        reasoning = short_reasoning if short_reasoning else "PLAN: Analyze query and respond directly."
        return f"### Prompt:\n{prompt}\n\n### Plan:\n{reasoning}\n\n### Response:\n{final_answer}"
        
    elif mode == "draft_critique_revision":
        c_draft = draft if draft else "Draft response placeholder."
        c_crit = critique if critique else "Critique placeholder."
        c_rev = revision if revision else final_answer
        return f"### Prompt:\n{prompt}\n\n### Draft:\n{c_draft}\n\n### Critique:\n{c_crit}\n\n### Response:\n{c_rev}"
        
    elif mode == "principles_plan_final":
        principles = format_principles(principles_list)
        reasoning = short_reasoning if short_reasoning else "PLAN: Analyze query and respond directly."
        return f"### Prompt:\n{prompt}\n\n### Principles:\n{principles}\n\n### Plan:\n{reasoning}\n\n### Response:\n{final_answer}"
        
    else:
        return f"### Prompt:\n{prompt}\n\n### Response:\n{final_answer}"


class ConstitutionalDataset(IterableDataset):
    """
    Reads JSONL files containing constitutional records, formats them
    using a selected template mode, tokenizes them, and yields blocks
    of shape (block_size,).
    """
    def __init__(self, source: str, tokenizer_name: str = "gpt2", block_size: int = 256, 
                 template_mode: str = "final_only", split: str = "train"):
        super().__init__()
        self.source = source
        self.tokenizer_name = tokenizer_name
        self.block_size = block_size
        self.template_mode = template_mode
        self.split = split
        
        try:
            self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        except Exception:
            self.tokenizer = tiktoken.get_encoding("gpt2")
            
        self.is_hf = not source.endswith(".jsonl") and not os.path.exists(source) and "/" in source
        self.has_val_split = True
        if self.is_hf:
            from datasets import get_dataset_split_names
            try:
                available_splits = get_dataset_split_names(source)
                self.has_val_split = any(s in ["validation", "val"] for s in available_splits)
            except Exception:
                self.has_val_split = True

    def __iter__(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        if self.is_hf:
            from datasets import load_dataset
            actual_split = self.split
            if self.split in ["val", "validation"] and not self.has_val_split:
                actual_split = "train"
                
            try:
                dataset = load_dataset(self.source, split=actual_split, streaming=True)
            except Exception as e:
                if self.split in ["val", "validation"]:
                    alt_split = "validation" if self.split == "val" else "val"
                    try:
                        dataset = load_dataset(self.source, split=alt_split, streaming=True)
                        self.has_val_split = True
                    except Exception:
                        try:
                            dataset = load_dataset(self.source, split="train", streaming=True)
                            self.has_val_split = False
                        except Exception:
                            raise e
                else:
                    raise e
            records = dataset
        else:
            file_paths = glob.glob(self.source)
            file_paths.sort()
            if not file_paths:
                raise FileNotFoundError(f"No JSONL files found matching source: {self.source}")
            
            if len(file_paths) > 1:
                split_idx = int(len(file_paths) * 0.9)
                active_files = file_paths[:split_idx] if self.split == "train" else file_paths[split_idx:]
                if not active_files:
                    active_files = file_paths
            else:
                active_files = file_paths
            
            def local_records_generator():
                block_count = 0
                for file_path in active_files:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                record = json.loads(line)
                            except Exception:
                                continue
                            
                            if len(file_paths) == 1:
                                is_val = (block_count % 10 == 9) or (block_count < 10 and block_count % 2 == 1)
                                block_count += 1
                                if (self.split == "val" and not is_val) or (self.split == "train" and is_val):
                                    continue
                            
                            yield record
            records = local_records_generator()

        token_buffer = []
        eos_token_id = self.tokenizer.eot_token
        worker_info = torch.utils.data.get_worker_info()
        
        record_idx = -1
        for raw_record_idx, record in enumerate(records):
            if self.is_hf and not self.has_val_split:
                is_val_sample = (raw_record_idx % 10 == 9)
                if self.split in ["val", "validation"] and not is_val_sample:
                    continue
                if self.split == "train" and is_val_sample:
                    continue
                    
            record_idx += 1
            if worker_info is not None:
                if record_idx % worker_info.num_workers != worker_info.id:
                    continue
            
            text = format_constitutional_record(record, self.template_mode)
            tokens = self.tokenizer.encode(text, allowed_special="all")
            tokens.append(eos_token_id)
            token_buffer.extend(tokens)
            
            while len(token_buffer) >= self.block_size + 1:
                chunk = token_buffer[:self.block_size + 1]
                token_buffer = token_buffer[self.block_size:]
                
                x = np.array(chunk[:-1], dtype=np.int64)
                y = np.array(chunk[1:], dtype=np.int64)
                yield x, y

        if len(token_buffer) > 0:
            padding_len = (self.block_size + 1) - len(token_buffer)
            if padding_len > 0:
                token_buffer.extend([eos_token_id] * padding_len)
            chunk = token_buffer[:self.block_size + 1]
            x = np.array(chunk[:-1], dtype=np.int64)
            y = np.array(chunk[1:], dtype=np.int64)
            yield x, y
