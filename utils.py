import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer


def encode_df(tokenizer, df):
	sources_encodings = []
	mask_encodings = []
	targets_encodings = []

	for i in tqdm(range(len(df))):
	    correct = df['target'][i]
	    erroneous = df['source'][i]
	    # print(correct)
	    # print(erroneous)
	    correct_encoding = tokenizer(correct)
	    erroneous_encoding = tokenizer(erroneous)
	    # print(correct_encoding)
	    # print(erroneous_encoding)
	    source_encoding = erroneous_encoding['input_ids']
	    mask_encoding = erroneous_encoding['attention_mask']
	    target_encoding = correct_encoding['input_ids']
	    # print(train_source_encoding)
	    # print(train_mask_encoding)
	    # print(train_target_encoding)
	    sources_encodings.append(source_encoding)
	    mask_encodings.append(mask_encoding)
	    targets_encodings.append(target_encoding)

	return (sources_encodings, mask_encodings, targets_encodings)