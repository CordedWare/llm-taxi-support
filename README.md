# Support AI Pet-Project

Проект для обучения модели T5 на паре «вопрос-ответ» для поддержки такси.

## Установка

```bash
python -m venv venv
source venv/bin/activate  
deactivate # 
pip install -r requirements.txt
```


## Обучение модели
```bash
python train.py # Запуск обучения
python infer.py # Запуск интерактивного режима генерации ответов
Добавление данных:
Файл data/dataset.json содержит пары вопрос-ответ.
Добавляй новые объекты с "input_text" и "target_text".
```
## Пример как работает обученная модель
<img width="1280" height="243" alt="image" src="https://github.com/user-attachments/assets/9138bf4b-3bd5-4d76-b646-e3db5fab16c7" />
