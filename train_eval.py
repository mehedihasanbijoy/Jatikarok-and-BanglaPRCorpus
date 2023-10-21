from tqdm import tqdm


def train(N_EPOCHS, model, train_loader, optim, device):
	model.train()
	print("training has started")
	for epoch in range(epoch, N_EPOCHS):
	    print(f"Epoch = {epoch}")
	    epoch_loss = 0
	    model.train()

	    for (input_ids, attention_mask, target_ids) in tqdm(train_loader):
	        input_ids = input_ids.to(device)
	        attention_mask = attention_mask.to(device)
	        target_ids = target_ids.to(device)

	        optim.zero_grad()
	        predictions = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
	        loss = predictions[0]
	        loss.backward()
	        epoch_loss += loss.item()
	        optim.step()

	    epoch_loss = epoch_loss/len(train_loader)
	    print(f"Loss = {epoch_loss}")
	    return epoch_loss
	# Training Loop Ends Here


def evaluation(model, test_loader, tokenizer, device):
	print("evaluation has started")
	model.eval()
	all_preds = []
	true_corrections = []
	pred_outputs = []

	for (input_ids, attention_mask, target_ids) in test_loader:
	    input_ids = input_ids.to(device)
	    attention_mask = attention_mask.to(device)
	    target_ids = target_ids.to(device)

	    predictions = model.generate(input_ids=input_ids, attention_mask=attention_mask)
	    # predictions = model.generate(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
	    # print(predictions.shape, target_ids.shape)

	    trg_text = [tokenizer.decode(token, skip_special_tokens=True) for token in target_ids]
	    prd_text = [tokenizer.decode(token, skip_special_tokens=True).replace('<extra_id_-25912>', '')[1:] for token in predictions]
	    # prd_text = [' '.join(tokenizer.decode(token, skip_special_tokens=True).split()[1:]) for token in predictions]
	    # prd_text = [' '.join(tokenizer.decode(token, skip_special_tokens=True).split()[1:]) for token in predictions]
	    # print(prd_text)
	    # print(trg_text)

	    true_corrections += trg_text
	    pred_outputs += prd_text

	    # all_preds.extend([x == y for x, y in zip(prd_text, trg_text)])

	    # predictions = predictions[1]
	    # print(torch.argmax(predictions, dim= -1).shape)
	# print(f"Accuracy: {sum(all_preds) / len(all_preds) * 100 : .2f}%")
	return true_corrections, pred_outputs