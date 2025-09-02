import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from config import *
from logger import get_logger

logger = get_logger(__name__)

def generate_answer(query):

    device    = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained("support_model")
    model     = T5ForConditionalGeneration.from_pretrained("support_model")
    model.to(device)
    model.eval()

    inputs = tokenizer(query, return_tensors = "pt", truncation = True, padding = True, max_length = MAX_LENGTH).to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_length=MAX_LENGTH)
    answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    while True:
        user_input = input("Введите вопрос: ")
        if user_input.lower() in ["выход", "exit", "quit"]:
            break
        response = generate_answer(user_input)
        print("Ответ модели:", response)
