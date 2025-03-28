from data_utils import SeparatedMelHarmTextDataset, MelHarmTextCollatorForSeq2Seq
import os
import numpy as np
from harmony_tokenizers_m21 import ChordSymbolTokenizer, RootTypeTokenizer, \
    PitchClassTokenizer, RootPCTokenizer, GCTRootPCTokenizer, \
    GCTSymbolTokenizer, GCTRootTypeTokenizer, MelodyPitchTokenizer, \
    MergedMelHarmTokenizer
from torch.utils.data import DataLoader
from transformers import BartForConditionalGeneration, BartConfig, DataCollatorForSeq2Seq
import torch
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn as nn
import csv
import argparse
# from transformers import RobertaModel, RobertaTokenizer

from models import TransTextVAE

tokenizers = {
    'ChordSymbolTokenizer': ChordSymbolTokenizer,
    'RootTypeTokenizer': RootTypeTokenizer,
    'PitchClassTokenizer': PitchClassTokenizer,
    'RootPCTokenizer': RootPCTokenizer,
    'GCTRootPCTokenizer': GCTRootPCTokenizer,
    'GCTSymbolTokenizer': GCTSymbolTokenizer,
    'GCTRootTypeTokenizer': GCTRootTypeTokenizer
}

description_modes = [
    'specific_chord',
    'chord_root',
    'pitch_class'
]

def main():

# train_dir = '/mnt/ssd2/maximos/data/hooktheory_train'
# test_dir = '/mnt/ssd2/maximos/data/hooktheory_test'

# Create the argument parser
    parser = argparse.ArgumentParser(description='Script for training BART model with a specific harmonic tokenizer.')

    # Define arguments
    parser.add_argument('-t', '--tokenizer', type=str, help='Specify the tokenizer name among: ' + repr(tokenizers.keys()), required=True)
    parser.add_argument('-m', '--description_mode', type=str, help='Specify the description mode name among: ' + repr(description_modes), required=True)
    parser.add_argument('-d', '--datatrain', type=str, help='Specify the full path to the root folder of the training xml/mxl files', required=True)
    parser.add_argument('-v', '--dataval', type=str, help='Specify the full path to the root folder of the validation xml/mxl files', required=True)
    parser.add_argument('-g', '--gpu', type=int, help='Specify whether and which GPU will be used by used by index. Not using this argument means use CPU.', required=False)
    parser.add_argument('-e', '--epochs', type=int, help='Specify number of epochs. Defaults to 100.', required=False)
    parser.add_argument('-l', '--learningrate', type=float, help='Specify learning rate. Defaults to 5e-5.', required=False)
    parser.add_argument('-b', '--batchsize', type=int, help='Specify batch size. Defaults to 16.', required=False)
    
    # Parse the arguments
    args = parser.parse_args()
    tokenizer_name = args.tokenizer
    description_mode = args.description_mode
    # root_dir = '/media/maindisk/maximos/data/hooktheory_xmls'
    train_dir = args.datatrain
    val_dir = args.dataval
    device_name = 'cpu'
    if args.gpu is not None:
        if args.gpu > -1:
            device_name = 'cuda:' + str(args.gpu)
    epochs = 1000
    if args.epochs:
        epochs = args.epochs
    lr = 5e-5
    if args.learningrate:
        lr = args.learningrate
    batchsize = 16
    if args.batchsize:
        batchsize = args.batchsize

    melody_tokenizer = MelodyPitchTokenizer.from_pretrained('saved_tokenizers/MelodyPitchTokenizer')
    harmony_tokenizer = tokenizers[tokenizer_name].from_pretrained('saved_tokenizers/' + tokenizer_name)

    tokenizer = MergedMelHarmTokenizer(melody_tokenizer, harmony_tokenizer)

    train_dataset = SeparatedMelHarmTextDataset(train_dir, tokenizer, max_length=512, num_bars=64, \
        description_mode=description_mode)
    test_dataset = SeparatedMelHarmTextDataset(val_dir, tokenizer, max_length=512, num_bars=64, \
        description_mode=description_mode)

    if device_name == 'cpu':
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            print('Selected device not available: ' + device_name)

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

    def create_data_collator(tokenizer, model):
        return MelHarmTextCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    # end create_data_collator

    bart_path = 'saved_models/bart/' + tokenizer_name + '/' + tokenizer_name + '.pt'
    checkpoint = torch.load(bart_path, map_location=device_name, weights_only=True)
    bart.load_state_dict(checkpoint)

    for param in bart.parameters():
        param.requires_grad = False

    collator = create_data_collator(tokenizer, model=bart)
    trainloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, collate_fn=collator)
    valloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, collate_fn=collator)

    config = {
        'lstm_dim': 2048,
        'roberta_model': "roberta-base",
        'latent_dim': 2048,
        'freeze_roberta': True
    }
    
    model = TransTextVAE(bart, tokenizer=tokenizer, device=device, config=config)
    model.to(device)

    model.cvae.train()
    optimizer = AdamW(model.cvae.parameters(), lr=lr)

    # save results
    os.makedirs('results/bart_text_cvae/'+tokenizer_name+'/', exist_ok=True)
    results_path = 'results/bart_text_cvae/' + tokenizer_name+'/'+description_mode + '.csv'
    result_fields = ['epoch', 'train_loss', 'val_loss', 'sav_version']
    with open( results_path, 'w' ) as f:
        writer = csv.writer(f)
        writer.writerow( result_fields )

    # keep best validation loss for saving
    best_val_loss = np.inf
    save_dir = 'saved_models/bart_text_cvae/' + tokenizer_name+'/'+description_mode + '/'
    os.makedirs(save_dir, exist_ok=True)
    transformer_cvae_path = save_dir + tokenizer_name + '_' + description_mode + '.pt'
    saving_version = 0

    # Training loop
    for epoch in range(epochs):
        print('training')
        train_loss = 0
        running_loss = 0
        batch_num = 0
        with tqdm(trainloader, unit='batch') as tepoch:
            tepoch.set_description(f'Epoch {epoch} | trn')
            for batch in tepoch:
                input_ids = batch['input_ids'].to(device)
                txts = batch['txt']
                # attention_mask = batch['attention_mask'].to(device)
                # labels = batch['labels'].to(device)
                outputs = model(input_ids, txts)
                loss = outputs['loss']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update loss
                batch_num += 1
                running_loss += loss.item()
                train_loss = running_loss/batch_num
                tepoch.set_postfix(loss=train_loss)#, accuracy=0)
        val_loss = 0
        running_loss = 0
        batch_num = 0
        print('validation')
        with torch.no_grad():
            with tqdm(valloader, unit='batch') as tepoch:
                tepoch.set_description(f'Epoch {epoch} | val')
                for batch in tepoch:
                    input_ids = batch['input_ids'].to(device)
                    txts = batch['txt']
                    # attention_mask = batch['attention_mask'].to(device)
                    # labels = batch['labels'].to(device)
                    outputs = model(input_ids, txts)
                    loss = outputs['loss']

                    # update loss
                    batch_num += 1
                    running_loss += loss.item()
                    val_loss = running_loss/batch_num
                    tepoch.set_postfix(loss=val_loss)#, accuracy=0)
        if best_val_loss > val_loss:
            print('saving!')
            saving_version += 1
            best_val_loss = val_loss
            torch.save(model.state_dict(), transformer_cvae_path)
            print(f'validation: loss={val_loss}')
        with open( results_path, 'a' ) as f:
            writer = csv.writer(f)
            writer.writerow( [epoch, train_loss, val_loss, saving_version] )

# end main

if __name__ == '__main__':
    main()