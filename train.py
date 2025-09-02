import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader
from config import *
from dataset import SupportDataset
from logger import get_logger

logger = get_logger(__name__)
 
def train():
    
    device     = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    tokenizer  = T5Tokenizer.from_pretrained(MODEL_NAME)
    model      = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(device)
    dataset    = SupportDataset(DATASET_PATH, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)
    optimizer  = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE)

    model.train()
    total_batches = len(dataloader)
    for epoch in range(EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{EPOCHS} started")
        for batch_idx, batch in enumerate(dataloader, start = 1):
            optimizer.zero_grad()
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)
            loss    = outputs.loss
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0 or batch_idx == total_batches:
                logger.info(f"Epoch {epoch + 1}/{EPOCHS} - Batch {batch_idx}/{total_batches} - Loss: {loss.item():.4f}")

    # Сохраняю модельку после обучения
    model.save_pretrained("support_model")
    tokenizer.save_pretrained("support_model")
    logger.info("Model saved to support_model/")

if __name__ == "__main__":
    train()
