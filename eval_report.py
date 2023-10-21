from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score


def evaluation_report(true_corrections, pred_outputs):
	acc = accuracy_score(y_true=true_corrections, y_pred=pred_outputs)
	pr = precision_score(y_true=true_corrections, y_pred=pred_outputs, average='micro')
	re = recall_score(y_true=true_corrections, y_pred=pred_outputs, average='micro')
	# f1 = f1_score(y_true=true_corrections, y_pred=pred_outputs, average='micro')
	f1 = fbeta_score(y_true=true_corrections, y_pred=pred_outputs, average='micro', beta=1)
	f05 = fbeta_score(y_true=true_corrections, y_pred=pred_outputs, average='micro', beta=0.5)

	print(f"Accuracy Score = {acc*100:.2f}%")
	print(f"Precision Score = {pr:.5f}")
	print(f"Recall Score = {re:.5f}")
	print(f"F1 Score = {f1:.5f}")
	print(f"F0.5 Score = {f05:.5f}")