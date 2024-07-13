# Text Summarization Project using T5-small

This project implements a text translation system using the T5-small model fine-tuned on the XSum dataset. The model architecture leverages the `AutoModelForSeq2SeqLMn` and `AutoTokenizer` classes from Hugging Face's `transformers` library, and the `Seq2SeqTrainer` for training.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/lakshayd760/Text_summarization.git
    cd text-translation-t5-small
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Dataset

This project uses the [XSum dataset](https://huggingface.co/datasets/xsum) from Hugging Face. The XSum dataset contains BBC articles accompanied by single-sentence summaries. We use these summaries for the text translation task.

## Model Architecture

The model is based on the `T5-small` architecture. The key components include:
- `t5-small model`: Using T5-small model for tokenization and fine tuninig
- `AutoModelForSeq2SeqLM`: The core model for conditional generation tasks.
- `AutoTokenizer`: The tokenizer specific to the model.
- `Seq2SeqTrainer`: The trainer for sequence-to-sequence tasks.

## Training

To fine-tune the T5-small model on the XSum dataset, follow these steps:

1. Prepare the dataset:
    ```python
    from datasets import load_dataset
    dataset = load_dataset("xsum")
    ```

2. Initialize the model and tokenizer:
    ```python
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLMn.from_pretrained(model_name)
    ```

3. Tokenize the dataset:
    ```python
    def preprocess_function(examples):
        inputs = [ex["document"] for ex in examples["train"]]
        targets = [ex["summary"] for ex in examples["train"]]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)
    ```

4. Set up the training arguments and trainer:
    ```python
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    training_args = Seq2SeqTrainingArguments(
        model_name,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    ```

## Evaluation

To evaluate the fine-tuned model, you can use the following code:
```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result
```
## Usage

After training, you can use the fine-tuned model for text translation:

```python
def translate(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

translated_text = translate("Your input text here.")
print(translated_text)
```
