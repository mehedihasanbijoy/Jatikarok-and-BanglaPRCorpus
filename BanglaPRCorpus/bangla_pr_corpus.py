import argparse
import pandas as pd

from corpus_utils import {
	read_train_valid_test_dfs, 
	find_unique_characters, 
	find_unique_bangla_characters, 
	clean_sentences, 
	remove_N_punctuation_mark
}





def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--TRAIN_PATH", help="Path of the csv file of training set", type=str, default="./BanglaParaphraseCorpus/train.csv", 
        choices=[
            "./BanglaParaphraseCorpus/train.csv", "./BanglaParaphraseCorpus/valid.csv", "./BanglaParaphraseCorpus/test.csv"
        ]
    )
    parser.add_argument("--VALID_PATH", help="Path of the csv file of validation set", type=str, default="./BanglaParaphraseCorpus/valid.csv", 
        choices=[
            "./BanglaParaphraseCorpus/train.csv", "./BanglaParaphraseCorpus/valid.csv", "./BanglaParaphraseCorpus/test.csv"
        ]
    )
    parser.add_argument("--TEST_PATH", help="Path of the csv file of test set", type=str, default="./BanglaParaphraseCorpus/test.csv", 
        choices=[
            "./BanglaParaphraseCorpus/train.csv", "./BanglaParaphraseCorpus/valid.csv", "./BanglaParaphraseCorpus/test.csv"
        ]
    )
    parser.add_argument("--SAVE_AT", help="Path of the newly created BanglaPRCorpus path", type=str, default="./BanglaPRCorpus/corpus.csv", 
        choices=[
            "./BanglaPRCorpus/corpus.csv"
        ]
    )
    args = parser.parse_args()

    train_df, validation_df, test_df = read_train_valid_test_dfs(args.TRAIN_PATH, args.VALID_PATH, args.TEST_PATH)

    temp_df = pd.concat([train_df, test_df, valid_df], axis = 0)

    unique_characters = find_unique_characters(source_data = temp_df['source'].values)
    list_of_unique_bangla_chars = find_unique_bangla_characters(list_of_characters = unique_characters)

    # list of Bangla punctuation marks have been considered in the paper
    bangla_punctuations = [
	    '।', ',', '!', '?', ';', 'ঃ', ':', '“', '‘', '-', '(', ')', '{', '}', '[', ']'
	]

	# list of all considered characters frequently appeared in Bangla text including a space
	all_considered_characters = list_of_unique_bangla_chars + bangla_punctuations + [' ']

	all_sents = list(temp_df['source'].values) + list(temp_df['target'].values)

	cleaned_sents = clean_sentences(list_of_sentences = all_sents, list_of_all_considered_characters = all_considered_characters)

	main_df = pd.DataFrame()

	for idx in range(1, 10+1):
		print(f"{idx} punctuation mark is being removed from the sentences")
		sources, targets, nopr = remove_N_punctuation_mark(N = idx, list_of_punctuation_mark = bangla_punctuations, list_of_sentences = cleaned_sents)
		df = pd.DataFrame({'source': sources, 'target': targets, 'nopr': nopr})
		main_df.concat([main_df, df], axis = 0)
		print(f"{idx} punctuation mark is removed from the sentences \n\n\n")

	main_df.reset_index(drop=True, inplace=True)

	main_df.to_csv(args.SAVE_AT, index = False)





if __name__ == "__main__":
	main()