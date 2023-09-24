import pandas as pd
from tqdm import tqdm
import random





def read_train_valid_test_dfs(train_df_path=None, valid_df_path=None, test_df_path=None):
	if train_df_path==None or valid_df_path==None or test_df_path==None:
		raise "\tTrain/valid/test path has not been defined which is required to proceed\n"

	train_df = pd.read_csv(train_df_path)
	valid_df = pd.read_csv(valid_df_path)
	test_df = pd.read_csv(test_df_path)

	return (train_df, valid_df, test_df)





def find_unique_characters(source_data):
	unique_characters = []

	for sent in tqdm(source_data):
		for char in sent:
			if char not in unique_characters:
				unique_characters.append(char)

	return unique_characters





def find_unique_bangla_characters(list_of_characters):
	list_of_unique_bangla_chars = []

	for char in list_of_characters:
		if char >= 'ঀ' and char <= '৾':
			list_of_unique_bangla_chars.append(char)

	return list(set(list_of_unique_bangla_chars))





def clean_sentences(list_of_sentences, list_of_all_considered_characters):
	cleaned_sents = []

	for sent in tqdm(list_of_sentences):
		_sent = []
		for char in sent:
			if char in list_of_all_considered_characters:
				_sent.append(char)
			else:
				_sent.append('')

		cleaned_sents.append(''.join(_sent))

	return cleaned_sents





def remove_N_punctuation_mark(N, list_of_punctuation_mark, list_of_sentences):
	sources, targets, nopr = [], [], []

	for sent in tqdm(list_of_sentences):
		count = 0
		for char in sent:
			if char in list_of_punctuation_mark:
				count += 1

		if count < N:
			continue

		targets.append(sent)

		for i in range(N):
			random.shuffle(list_of_punctuation_mark)
			for punctuation in list_of_punctuation_mark:
				if punctuation in sent:
					sent = sent.replace(punctuation, '')
					break

		sources.append(sent)
		nopr.append(N)

	return (sources, targets, nopr)





if __name__ == "__main__":
	pass