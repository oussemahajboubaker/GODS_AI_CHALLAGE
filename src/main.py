import pandas as pd
from tqdm import tqdm

# DL imports
import torch
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModel

# ML imports
from sklearn.metrics import accuracy_score

tqdm.pandas()

DATA_PATH = ""
MODEL_NAME = "mental/mental-bert-base-uncased"
MAX_LEN = 512
BATCH_SIZE = 32
LEARNING_RATE = 2e-5  # 2e-5, 3e-5, 5e-5
EPOCHS = 10
NUM_CLASSES = 5
# PATIENCE_ES = 5, not used
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
N_RUN = 1  # 3


def main():
    # Load data
    train = pd.read_csv(DATA_PATH + "train.csv")
    test = pd.read_csv(DATA_PATH + "test.csv")
    val = pd.read_csv(DATA_PATH + "val.csv")

    # load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME).to(device)
    model = AutoModel.from_pretrained(MODEL_NAME, num_labels=NUM_CLASSES).to(device)


    # Tokenize data
    train_encodings = tokenizer(train.text.tolist(), truncation=True, padding=True, max_length=MAX_LEN)
    val_encodings = tokenizer(val.text.tolist(), truncation=True, padding=True, max_length=MAX_LEN)
    test_encodings = tokenizer(test.text.tolist(), truncation=True, padding=True, max_length=MAX_LEN)

    args = TrainingArguments(
        output_dir="mental-bert-base-uncased",
        evaluation_strategy = "epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
    )


    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}
    
    # Train model
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_encodings,
        eval_dataset=val_encodings,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Evaluate model
    trainer.evaluate(test_encodings)


    # Save model
    model.save_pretrained("mental-bert-base-uncased")
    tokenizer.save_pretrained("mental-bert-base-uncased")



if __name__ == "__main__":
    main()