import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
# ChordSymbolTokenizer is just for generating template-based descriptions
from harmony_tokenizers_m21 import MergedMelHarmTokenizer, ChordSymbolTokenizer
import random
import os
import numpy as np
from transformers import DataCollatorForSeq2Seq
import mir_eval
from copy import deepcopy

class SeparatedMelHarmTextDataset(Dataset):
    def __init__(self, root_dir, merged_tokenizer, max_length=512, pad_to_max_length=True, \
                return_attention_mask=False, num_bars=8, description_mode='specific_chord'):
        # root_dir: the directory that includes subdirectories with mlx or xml files
        # Walk through all subdirectories and files
        # description_modes: 'chord_root', 'specific_chord' (root+type), 'pitch_class'
        self.data_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith('.xml') or file.endswith('.mxl'):
                    full_path = os.path.join(dirpath, file)
                    self.data_files.append(full_path)
        self.merged_tokenizer = merged_tokenizer
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length
        self.num_bars = num_bars
        self.return_attention_mask = return_attention_mask
        self.description_mode = description_mode
    # end init

    def __len__(self):
        return len(self.data_files)
    # end len

    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        # adjust number of bars based no maximum length
        if self.max_length is not None and self.num_bars is not None:
            tmp_encoded_len = self.max_length + 1
            curr_num_bars = self.num_bars
            while tmp_encoded_len > self.max_length:
                encoded = self.merged_tokenizer.encode(data_file, max_length=self.max_length,\
                                pad_to_max_length=self.pad_to_max_length, num_bars=curr_num_bars)
                tmp_encoded_len = len(encoded['input_ids'])
                curr_num_bars -= 1
        else:
            encoded = self.merged_tokenizer.encode(data_file, max_length=self.max_length,\
                            pad_to_max_length=self.pad_to_max_length, num_bars=self.num_bars)
        # separate melody from harmony
        labels = torch.tensor(encoded['input_ids']).clone()
        start_harmony_position = np.where( np.array(encoded['input_ids']) == self.merged_tokenizer.vocab[self.merged_tokenizer.harmony_tokenizer.start_harmony_token] )[0][0]
        input_ids = torch.tensor(encoded['input_ids'][:start_harmony_position], dtype=torch.long)
        attention_mask = torch.tensor(encoded['attention_mask'][:start_harmony_position], dtype=torch.long)
        labels = labels[start_harmony_position:]  # Ignore question tokens and <h> in loss computation
        labels[ labels == self.merged_tokenizer.pad_token_id ] = -100
        # make description of specific (e.g., the third) bar
        txt = self.merged_tokenizer.make_description_of_tokens_list_at_random_bar(
            encoded['input_tokens'][start_harmony_position:],
            self.description_mode
        )
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'txt': txt
        }
    # end getitem
# end class dataset

class MelHarmTextCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, model, padding=True):
        super().__init__(tokenizer=tokenizer, model=model, padding=padding)
    # end init

    def __call__(self, features):
        standard_features = [
            {k: v for k, v in f.items() if k in ["input_ids", "attention_mask", "labels"]}
            for f in features
        ]
        # Process main tokenizer fields
        batch = super().__call__(standard_features)

        # Extract raw text from dataset
        txt_strings = [f["txt"] for f in features]
        batch['txt'] = txt_strings

        return batch
    # end call
# end MelHarmTextCollatorForSeq2Seq

class SeparatedMelHarmMarkovDataset(Dataset):
    def __init__(self, root_dir, merged_tokenizer, max_length=512, pad_to_max_length=True, \
                return_attention_mask=False, num_bars=8):
        # root_dir: the directory that includes subdirectories with mlx or xml files
        # Walk through all subdirectories and files
        self.data_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith('.xml') or file.endswith('.mxl'):
                    full_path = os.path.join(dirpath, file)
                    self.data_files.append(full_path)
        self.merged_tokenizer = merged_tokenizer
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length
        self.num_bars = num_bars
        self.return_attention_mask = return_attention_mask
    # end init

    def __len__(self):
        return len(self.data_files)
    # end len

    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        # adjust number of bars based no maximum length
        if self.max_length is not None and self.num_bars is not None:
            tmp_encoded_len = self.max_length + 1
            curr_num_bars = self.num_bars
            while tmp_encoded_len > self.max_length:
                encoded = self.merged_tokenizer.encode(data_file, max_length=self.max_length,\
                                pad_to_max_length=self.pad_to_max_length, num_bars=curr_num_bars)
                tmp_encoded_len = len(encoded['input_ids'])
                curr_num_bars -= 1
        else:
            encoded = self.merged_tokenizer.encode(data_file, max_length=self.max_length,\
                            pad_to_max_length=self.pad_to_max_length, num_bars=self.num_bars)
        # separate melody from harmony
        labels = torch.tensor(encoded['input_ids']).clone()
        start_harmony_position = np.where( np.array(encoded['input_ids']) == self.merged_tokenizer.vocab[self.merged_tokenizer.harmony_tokenizer.start_harmony_token] )[0][0]
        input_ids = torch.tensor(encoded['input_ids'][:start_harmony_position], dtype=torch.long)
        attention_mask = torch.tensor(encoded['attention_mask'][:start_harmony_position], dtype=torch.long)
        labels = labels[start_harmony_position:]  # Ignore question tokens and <h> in loss computation
        labels[ labels == self.merged_tokenizer.pad_token_id ] = -100
        # make mir_eval transition table
        transitions = self.merged_tokenizer.make_markov_from_tokens_list(encoded['input_tokens'][start_harmony_position:])
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'transitions': transitions
        }
    # end getitem
# end class dataset

class MergedMelHarmDataset(Dataset):
    def __init__(self, root_dir, merged_tokenizer, max_length=512, pad_to_max_length=True, \
                return_attention_mask=False, return_harmonization_labels=False,\
                num_bars=8):
        # root_dir: the directory that includes subdirectories with mlx or xml files
        # Walk through all subdirectories and files
        self.data_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith('.xml') or file.endswith('.mxl'):
                    full_path = os.path.join(dirpath, file)
                    self.data_files.append(full_path)
        self.merged_tokenizer = merged_tokenizer
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length
        self.num_bars = num_bars
        self.return_attention_mask = return_attention_mask
        self.return_harmonization_labels = return_harmonization_labels
    # end init

    def __len__(self):
        return len(self.data_files)
    # end len

    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        # adjust number of bars based no maximum length
        if self.max_length is not None and self.num_bars is not None:
            tmp_encoded_len = self.max_length + 1
            curr_num_bars = self.num_bars
            while tmp_encoded_len > self.max_length:
                encoded = self.merged_tokenizer.encode(data_file, max_length=self.max_length,\
                                pad_to_max_length=self.pad_to_max_length, num_bars=curr_num_bars)
                tmp_encoded_len = len(encoded['input_ids'])
                curr_num_bars -= 1
        else:
            encoded = self.merged_tokenizer.encode(data_file, max_length=self.max_length,\
                            pad_to_max_length=self.pad_to_max_length, num_bars=self.num_bars)
        if self.return_harmonization_labels:
            input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long)
            attention_mask = torch.tensor(encoded['attention_mask'], dtype=torch.long)
            # Generate labels: mask the question part
            sep_token_idx = (input_ids == self.merged_tokenizer.vocab['<h>']).nonzero(as_tuple=True)[0]
            labels = input_ids.clone()
            labels[:sep_token_idx + 1] = -100  # Ignore question tokens and <h> in loss computation
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
        elif self.return_attention_mask:
            return {
                'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long)
            }
        else:
            return torch.tensor(encoded['input_ids'], dtype=torch.long)
    # end getitem
# end class dataset

class SeparatedMelHarmDataset(Dataset):
    def __init__(self, root_dir, merged_tokenizer, max_length=512, pad_to_max_length=True, \
                return_attention_mask=False, num_bars=8):
        # root_dir: the directory that includes subdirectories with mlx or xml files
        # Walk through all subdirectories and files
        self.data_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith('.xml') or file.endswith('.mxl'):
                    full_path = os.path.join(dirpath, file)
                    self.data_files.append(full_path)
        self.merged_tokenizer = merged_tokenizer
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length
        self.num_bars = num_bars
        self.return_attention_mask = return_attention_mask
    # end init

    def __len__(self):
        return len(self.data_files)
    # end len

    def __getitem__(self, idx):
        data_file = self.data_files[idx]
        # adjust number of bars based no maximum length
        if self.max_length is not None and self.num_bars is not None:
            tmp_encoded_len = self.max_length + 1
            curr_num_bars = self.num_bars
            while tmp_encoded_len > self.max_length:
                encoded = self.merged_tokenizer.encode(data_file, max_length=self.max_length,\
                                pad_to_max_length=self.pad_to_max_length, num_bars=curr_num_bars)
                tmp_encoded_len = len(encoded['input_ids'])
                curr_num_bars -= 1
        else:
            encoded = self.merged_tokenizer.encode(data_file, max_length=self.max_length,\
                            pad_to_max_length=self.pad_to_max_length, num_bars=self.num_bars)
        # separate melody from harmony
        labels = torch.tensor(encoded['input_ids']).clone()
        start_harmony_position = np.where( np.array(encoded['input_ids']) == self.merged_tokenizer.vocab[self.merged_tokenizer.harmony_tokenizer.start_harmony_token] )[0][0]
        input_ids = torch.tensor(encoded['input_ids'][:start_harmony_position], dtype=torch.long)
        attention_mask = torch.tensor(encoded['attention_mask'][:start_harmony_position], dtype=torch.long)
        labels = labels[start_harmony_position:]  # Ignore question tokens and <h> in loss computation
        labels[ labels == self.merged_tokenizer.pad_token_id ] = -100
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    # end getitem
# end class dataset

class MLMCollator:
    def __init__(self, tokenizer, mask_prob=0.15):
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.mask_token_id = tokenizer.vocab[tokenizer.mask_token]
    # end init

    def __call__(self, batch):
        input_ids = torch.stack(batch)
        labels = input_ids.clone()
        
        # Create mask
        rand = torch.rand(input_ids.shape)
        mask = (rand < self.mask_prob) & (input_ids != self.tokenizer.vocab[self.tokenizer.pad_token])
        
        # Apply mask
        for i in range(input_ids.shape[0]):
            mask_idx = torch.where(mask[i])[0]
            for idx in mask_idx:
                prob = random.random()
                if prob < 0.8:
                    input_ids[i, idx] = self.mask_token_id  # 80% <mask>
                elif prob < 0.9:
                    input_ids[i, idx] = random.randint(0, len(self.tokenizer.vocab) - 1)  # 10% random
                # 10% unchanged (do nothing)

        # Replace labels of non-mask tokens with -100
        labels[~mask] = -100
        
        return {"input_ids": input_ids, "labels": labels}
    # end call

# end class MLMCollator

class GenCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.vocab[tokenizer.pad_token]
    # end init

    def __call__(self, batch):
        input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=self.pad_token_id)
        attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
        labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
        # also neutralize all that come pre-padded from the dataset
        labels[ labels == self.pad_token_id ] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    # end call
# end class GenCollator

class PureGenCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.vocab[tokenizer.pad_token]
    # end init

    def __call__(self, batch):
        input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=self.pad_token_id)
        attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
        labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
        # also neutralize all that come pre-padded from the dataset
        labels[ labels == self.pad_token_id ] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    # end call
# end class PureGenCollator

class MaskedGenCollator:
    def __init__(self, tokenizer, mask_prob=0.2, bar_id=None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.vocab[tokenizer.pad_token]
        self.mask_prob = mask_prob
        self.mask_token_id = tokenizer.mask_token_id  # Mask token ID
        self.bar_id = bar_id
    # end init

    def __call__(self, batch):
        # Apply masking before padding
        masked_input_ids = []
        for item in batch:
            input_ids = item["input_ids"].clone()
            for i in range(len(input_ids)):
                if input_ids[i] != self.bar_id and random.random() < self.mask_prob:
                    input_ids[i] = self.mask_token_id  # Replace with <mask>
            masked_input_ids.append(input_ids)

        input_ids = pad_sequence(masked_input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
        labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
        # also neutralize all that come pre-padded from the dataset
        labels[ labels == self.pad_token_id ] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    # end call
# end class MaskedGenCollator

class MaskedDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, mask_prob=0.2, bar_id=None, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.mask_prob = mask_prob
        self.mask_token_id = tokenizer.mask_token_id
        self.bar_id = bar_id  # Token that should not be masked
    # end init

    def __call__(self, features):
        batch = super().__call__(features)  # Get the default behavior
        
        # Apply masking only to the encoder input (input_ids)
        masked_input_ids = batch["input_ids"].clone()
        for i in range(masked_input_ids.shape[0]):  # Iterate over batch
            for j in range(masked_input_ids.shape[1]):  # Iterate over sequence
                if masked_input_ids[i, j] != self.bar_id and random.random() < self.mask_prob:
                    masked_input_ids[i, j] = self.mask_token_id

        batch["input_ids"] = masked_input_ids  # Replace with masked version
        return batch
    # end call
# end class MaskedDataCollatorForSeq2Seq