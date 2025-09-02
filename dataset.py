import json
from torch.utils.data import Dataset

class SupportDataset(Dataset):

    def __init__(self, file_path, tokenizer, max_length):
        # Загрузка данных из JSON файла
        with open(file_path, 'r', encoding = 'utf-8') as f:
            self.data = json.load(f)  # ожидается список словарей с keys: input_text и target_text
        
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Токенизация вопроса (входа)
        input_enc = self.tokenizer(
            item['input_text'],
            max_length     = self.max_length,
            padding        = 'max_length',
            truncation     = True,
            return_tensors = 'pt'
        )
        
        # Токенизация ответа (цели)
        target_enc = self.tokenizer(
            item['target_text'],
            max_length     = self.max_length,
            padding        ='max_length',
            truncation     = True,
            return_tensors = 'pt'
        )

        labels = target_enc['input_ids']
        # Заменяем паддинги на -100, чтобы loss их игнорировал
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids':      input_enc['input_ids'].squeeze(0),        # Убираем лишнюю размерность batch=1
            'attention_mask': input_enc['attention_mask'].squeeze(0),
            'labels':         labels.squeeze(0)
        }
