from data_utils import SeparatedMelHarmMarkovDataset
import os
import numpy as np
from harmony_tokenizers_m21 import ChordSymbolTokenizer, RootTypeTokenizer, \
    PitchClassTokenizer, RootPCTokenizer, GCTRootPCTokenizer, \
    GCTSymbolTokenizer, GCTRootTypeTokenizer, MelodyPitchTokenizer, \
    MergedMelHarmTokenizer
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BartConfig, DataCollatorForSeq2Seq
from tqdm import tqdm
from models import TransGraphVAE

train_dir = '/mnt/ssd2/maximos/data/hooktheory_train'
test_dir = '/mnt/ssd2/maximos/data/hooktheory_test'

chordSymbolTokenizer = ChordSymbolTokenizer.from_pretrained('saved_tokenizers/ChordSymbolTokenizer')
# rootTypeTokenizer = RootTypeTokenizer.from_pretrained('saved_tokenizers/RootTypeTokenizer')
# pitchClassTokenizer = PitchClassTokenizer.from_pretrained('saved_tokenizers/PitchClassTokenizer')
# rootPCTokenizer = RootPCTokenizer.from_pretrained('saved_tokenizers/RootPCTokenizer')
melodyPitchTokenizer = MelodyPitchTokenizer.from_pretrained('saved_tokenizers/MelodyPitchTokenizer')

m_chordSymbolTokenizer = MergedMelHarmTokenizer(melodyPitchTokenizer, chordSymbolTokenizer)
# m_rootTypeTokenizer = MergedMelHarmTokenizer(melodyPitchTokenizer, rootTypeTokenizer)
# m_pitchClassTokenizer = MergedMelHarmTokenizer(melodyPitchTokenizer, pitchClassTokenizer)
# m_rootPCTokenizer = MergedMelHarmTokenizer(melodyPitchTokenizer, rootPCTokenizer)

tokenizer = m_chordSymbolTokenizer
tokenizer_name = 'ChordSymbolTokenizer'

train_dataset = SeparatedMelHarmMarkovDataset(train_dir, tokenizer, max_length=512, num_bars=64)
test_dataset = SeparatedMelHarmMarkovDataset(train_dir, tokenizer, max_length=512, num_bars=64)

# Data collator for BART
def create_data_collator(tokenizer, model):
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
# end create_data_collator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bart_config = BartConfig(
    vocab_size=len(tokenizer.vocab),
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    decoder_start_token_id=tokenizer.bos_token_id,
    forced_eos_token_id=tokenizer.eos_token_id,
    max_position_embeddings=512,
    encoder_layers=8,
    encoder_attention_heads=8,
    encoder_ffn_dim=512,
    decoder_layers=8,
    decoder_attention_heads=8,
    decoder_ffn_dim=512,
    d_model=512,
    encoder_layerdrop=0.3,
    decoder_layerdrop=0.3,
    dropout=0.3
)

bart = BartForConditionalGeneration(bart_config)

bart_path = 'saved_models/bart/' + tokenizer_name + '/' + tokenizer_name + '.pt'
if device == 'cpu':
    checkpoint = torch.load(bart_path, map_location="cpu", weights_only=True)
else:
    checkpoint = torch.load(bart_path, weights_only=True)
bart.load_state_dict(checkpoint)

# # bart.to(device)

# bart_encoder, bart_decoder = bart.get_encoder(), bart.get_decoder()
# bart_encoder.to(device)
# bart_decoder.to(device)

# # Freeze BART parameters
# for param in bart_encoder.parameters():
#     param.requires_grad = False
# for param in bart_encoder.parameters():
#     param.requires_grad = False

for param in bart.parameters():
    param.requires_grad = False

collator = create_data_collator(tokenizer, model=bart)
trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collator)
testloader = DataLoader(test_dataset, batch_size=128, shuffle=True, collate_fn=collator)

epochs = 100

config = {
    'hidden_dim_LSTM': 256,
    'hidden_dim_GNN': 256,
    'latent_dim': 256,
    'condition_dim': 128,
    'use_attention': False
}

model = TransGraphVAE(transformer=bart, **config)
model.to(device)

model.cvae.train()
optimizer = AdamW(model.cvae.parameters(), lr=5e-5)

# Training loop
for epoch in range(epochs):
    print('training')
    with tqdm(trainloader, unit='batch') as tepoch:
        tepoch.set_description(f'Epoch {epoch} | trn')
        for batch in tepoch:
            input_ids = batch['input_ids'].to(device)
            transitions = batch['transitions'].to(device)
            # attention_mask = batch['attention_mask'].to(device)
            # labels = batch['labels'].to(device)
            outputs = model(input_ids, transitions)
            loss = outputs['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=loss.item(), accuracy=0)