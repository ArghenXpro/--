# KGvsOther-finetuned

**Языковой детектор: Kyrgyz vs Other**

Ниже — краткая инструкция по развёртыванию и использованию дообученной модели с Gradio:

---

## Ссылки

* **Colab (проверить модель)**: [https://colab.research.google.com/drive/1p1K-xvwYjSMQSn577jGYl6tW2Xl4Ntie?usp=sharing](https://colab.research.google.com/drive/1p1K-xvwYjSMQSn577jGYl6tW2Xl4Ntie?usp=sharing)
* **Colab (блокнот основного кода)**: [https://colab.research.google.com/drive/1ElQgQavfPw8gtmjqgI7MaaXWi9QIvBcR?usp=sharing](https://colab.research.google.com/drive/1ElQgQavfPw8gtmjqgI7MaaXWi9QIvBcR?usp=sharing)
* **Colab (гибридный пример)**: [https://colab.research.google.com/drive/1Zmols9Q\_gtqgoAGS18sVl\_qKxjkZKic4?usp=sharing](https://colab.research.google.com/drive/1Zmols9Q_gtqgoAGS18sVl_qKxjkZKic4?usp=sharing)

---

## How to use

1. **Распакуйте модель**

   ```bash
   !unzip -o KGvsOther-finetuned.zip
   ```

2. **Установка зависимостей**

   ```bash
   # 0) Установка (при необходимости)
   !pip install -q gradio transformers torch langdetect
   ```

3. **Импорты и настройка**

   ```python
   # 1) Импорты
   import gradio as gr
   import torch
   import torch.nn.functional as F
   from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
   from langdetect import detect, DetectorFactory

   # для детерминированности
   DetectorFactory.seed = 0

   # 2) Загрузка Roberta
   MODEL_DIR = "/content/KGvsOther-finetuned"
   tokenizer = XLMRobertaTokenizer.from_pretrained(
       MODEL_DIR,
       local_files_only=True
   )
   model     = XLMRobertaForSequenceClassification.from_pretrained(
       MODEL_DIR,
       local_files_only=True
   )
   model.config.id2label = {0: "Kyrgyz", 1: "Other"}
   model.config.label2id = {"Kyrgyz": 0, "Other": 1}
   model.eval()
   ```

4. **Комбинированная функция классификации**

   ```python
   # 3) Комбинированная функция
   def classify(text):
       # Быстрый языковой детектор
       try:
           lang = detect(text)
       except:
           lang = "unknown"
       if lang != "ky":
           # Всё, что не 'ky' — сразу Other
           return {"Kyrgyz": 0.0, "Other": 1.0}, "Other (100.0%)"

       # Иначе запускаем Roberta
       enc = tokenizer(
           text,
           padding="max_length",
           truncation=True,
           max_length=128,
           return_tensors="pt"
       )
       with torch.no_grad():
           logits = model(**enc).logits.squeeze(0)
           probs  = F.softmax(logits, dim=-1).tolist()

       labels    = [model.config.id2label[i] for i in range(len(probs))]
       prob_dict = {labels[i]: probs[i] for i in range(len(probs))}
       top_idx   = int(torch.argmax(torch.tensor(probs)))
       return prob_dict, f"{labels[top_idx]} ({probs[top_idx]*100:.1f}%)"
   ```

5. **Gradio-интерфейс**

   ```python
   # 4) Gradio-интерфейс
   iface = gr.Interface(
       fn=classify,
       inputs=gr.Textbox(lines=4, placeholder="Введите текст…"),
       outputs=[gr.Label(label="Probabilities"), gr.Textbox(label="Prediction")],
       title="Kyrgyz vs Other (langdetect + Roberta)",
       description=(
         "Сначала langdetect определяет язык: "
         "если не 'ky' → сразу Other; иначе — уточняет Roberta."
       )
   )

   iface.launch(share=True)
   ```

---

**Готово!**
