import warnings
warnings.filterwarnings('ignore')

# ! pip install datasets transformers sacrebleu torch sentencepiece transformers[sentencepiece]

import os
import random
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AdamW
import datasets
import sys
from transformers import AutoTokenizer
import argparse
from data_loader import DataLoader, collate_fn
from train_eval import train, evaluate
from eval_report import evaluation_report
from utils import encode_df


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--CORPUS_PATH", help="Path of the csv file of corpus", type=str, default="/BanglaPRCorpus/BanglaPRCorpus/corpus.csv", 
        choices=[
            "/BanglaPRCorpus/BanglaPRCorpus/corpus.csv", 
            "/BanglaPRCorpus/BanglaOPUS/corpus.csv", 
            "/BanglaPRCorpus/ProthomAloBalanced/corpus.csv"
        ]
    )
    parser.add_argument("--KNOWLEDGE_PATH", help="Path of the knowledge pth file", type=str, default="/Knowledge/knowledge4jatikarok.pth", 
        choices=[
            "/Knowledge/knowledge4jatikarok.pth", # ~ saved_model_gecV4MarianMT.py from Panini
            "/Knowledge/knowledge4t5small.pth", 
            "/Knowledge/knowledge4banglat5.pth"
        ]
    )
    parser.add_argument("--CHECKPOINT_PATH", help="Path of where model checkpoint should be saved", type=str, default="/Checkpoint/ckp4jatikarok.pth", 
        choices=[
            "/Checkpoint/ckp4jatikarok.pth",
            "/Checkpoint/ckp4banglat5.pth",
            "/Checkpoint/ckp4t5small.pth"
        ]
    )
    parser.add_argument("--MODEL_NAME", help="Name of the model", type=str, default="jatikarok", 
        choices=[
            "jatikarok", 
            "banglat5", 
            "t5small"
        ]
    )
    parser.add_argument("--N_EPOCHS", help="Number of epochs", type=int, default=50)
    parser.add_argument("--BATCH_SIZE", help="Batch size", type=int, default=16)

    args = parser.parse_args()

    model_name = args.MODEL_NAME
    N_EPOCHS = args.N_EPOCHS
	epoch = 0
	loss = 10e9
	PATH = args.CHECKPOINT_PATH #'/checkpoints/PuncBanglaPuncFormer.pth'
	K_PATH = args.KNOWLEDGE_PATH #'/knowledge/saved_model_gecV4MarianMT.pth'
	b_size = args.BATCH_SIZE
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	df = pd.read_csv(args.CORPUS_PATH)

	erroneous_sources = list(df['source'].values)
	correct_targets = list(df['target'].values)

	train_df = pd.DataFrame()
	test_df = pd.DataFrame()

	for idx in range(10):
	    temp_df = df[df['nopr'] == idx]
	    temp_df.reset_index(drop=True, inplace=True)
	    x = int(len(temp_df)*.2)
	    train_df = pd.concat([train_df, temp_df.iloc[:len(temp_df)-x, :]], axis=0)
	    test_df = pd.concat([test_df, temp_df.iloc[len(temp_df)-x:, :]], axis=0)
	    train_df.reset_index(drop=True, inplace=True)
	    test_df.reset_index(drop=True, inplace=True)

	train_sources = list(train_df['source'].values)
	test_sources = list(test_df['source'].values)
	train_targets = list(train_df['target'].values)
	test_targets = list(test_df['target'].values)

	tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglat5_banglaparaphrase", use_fast=False)

	print("training instances are being tokenized")
	train_sources_encodings, train_mask_encodings, train_targets_encodings = encode_df(tokenizer, train_df)

	print("test instances are being tokenized")
	test_sources_encodings, test_mask_encodings, test_targets_encodings = encode_df(tokenizer, test_df)

	train_dataset = LoadDataset(train_sources_encodings, train_mask_encodings, train_targets_encodings)
	test_dataset = LoadDataset(test_sources_encodings, test_mask_encodings, test_targets_encodings)

	train_loader = DataLoader(train_dataset, batch_size=b_size, collate_fn=collate_fn, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=b_size, collate_fn=collate_fn, shuffle=False)
	print("training and test dataloaders are in action with collate fn")

	if model_name == 'jatikarok':
		model_checkpoint = 'Helsinki-NLP/opus-mt-NORTH_EU-NORTH_EU'  # MarianMT 298M
		model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

	elif model_name == 'banglat5':
		model_checkpoint = 'csebuetnlp/banglat5_banglaparaphrase'
		model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

	elif model_name == 't5small':
		model_checkpoint = 't5-small'  # 242M
		model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

	else:
		raise Exception("Choose the model among jatikarok, banglat5, and t5small")

	model.to(device)
	print("model is in gpu now")

	optim = AdamW(model.parameters(), lr=5e-5)

	print("transfering the knowledge")
	if os.path.exists(K_PATH):
	    checkpoint = torch.load(K_PATH)
	    model.load_state_dict(checkpoint['model_state_dict'])
	print("knowledge transfered")

	print("incorporating model checkpoint")
	if os.path.exists(PATH):
	    checkpoint = torch.load(PATH)
	    model.load_state_dict(checkpoint['model_state_dict'])
	    epoch = checkpoint['epoch']
	    loss = checkpoint['loss']
	print("incorporated model checkpoint")

	epoch_loss = train(N_EPOCHS, model, train_loader, optim, device)

	if epoch_loss < loss:
        loss = epoch_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss,
        }, PATH)
        print(f"{'-'*20}\nModel Saved at {PATH}\n{'-'*20}\n")

	print("incorporating model checkpoint before evaluation")
	if os.path.exists(PATH):
	    checkpoint = torch.load(PATH)
	    model.load_state_dict(checkpoint['model_state_dict'])
	    epoch = checkpoint['epoch']
	    loss = checkpoint['loss']
	print("incorporated model checkpoint")


	true_corrections, pred_outputs = evaluation(model, test_loader, tokenizer, device)

	evaluation_report(true_corrections, pred_outputs)


if __name__ == '__main__':
	main()