from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset

# 1. Load a pretrained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# 2. Load a dataset (SQuAD)
dataset = load_dataset("squad")

# 3. Tokenize the dataset
def preprocess(examples):
    return tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        padding="max_length",
        max_length=384,
    )

tokenized_dataset = dataset.map(preprocess, batched=True)

# 4. Define training settings
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
)

# 5. Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
)

# 6. Train the model
trainer.train()


from transformers import pipeline

qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

result = qa_pipeline({
    "context": "New Delhi is the capital of India.",
    "question": "What is the capital of India?"
})

print(result["answer"])  # 👉 "New Delhi"



# Install (only once)
# pip install transformers datasets

from transformers import pipeline

# 1. Sentiment Analysis
sentiment = pipeline("sentiment-analysis")
print("Sentiment:", sentiment("I love AI!"))
# 👉 [{'label': 'POSITIVE', 'score': 0.999}]

# 2. Question Answering
qa = pipeline("question-answering")
print("QA:", qa({
    "context": "The capital of India is New Delhi.",
    "question": "What is the capital of India?"
}))
# 👉 {'score': 0.98, 'start': 24, 'end': 32, 'answer': 'New Delhi'}

# 3. Text Summarization
summarizer = pipeline("summarization")
print("Summary:", summarizer("Hugging Face makes NLP simple and powerful to use."))
# 👉 [{'summary_text': 'Hugging Face makes NLP easy and powerful.'}]

# 4. Translation (English → French)
translator = pipeline("translation_en_to_fr")
print("Translation:", translator("Hugging Face is awesome!"))
# 👉 [{'translation_text': 'Hugging Face est génial!'}]

# 5. Text Generation
generator = pipeline("text-generation", model="gpt2")
print("Generated:", generator("Once upon a time", max_length=20, num_return_sequences=1))
# 👉 [{'generated_text': 'Once upon a time there was a little girl who lived in a village...'}]

# 6. Named Entity Recognition (NER)
ner = pipeline("ner", grouped_entities=True)
print("NER:", ner("Elon Musk founded SpaceX in California."))
# 👉 [{'entity_group': 'PER', 'score': 0.999, 'word': 'Elon Musk'}, 
#     {'entity_group': 'ORG', 'score': 0.998, 'word': 'SpaceX'}, 
#     {'entity_group': 'LOC', 'score': 0.997, 'word': 'California'}]



models for languages like tel and eng and so on

🔹 1. Hugging Face mT5

mT5 = Multilingual T5 model from Google.

It’s a transformer-based model trained on 101 languages, including many Indian ones (Hindi, Telugu, Tamil, etc.).

It’s open-source (free to use on Hugging Face).

You can fine-tune it for translation, summarization, Q&A, etc. across languages.

👉 Example use:
Translate Hindi → English, or Summarize a Telugu document.

🔹 2. IndicTrans

IndicTrans = An open-source neural machine translation model created by AI4Bharat (IIT Madras).

Specifically trained for Indian languages ↔ English (and between Indian languages).

Better for Indian contexts than general models like mT5 (since it’s fine-tuned on Indian data).

Also free on Hugging Face.

👉 Example use:
Telugu → Hindi, Kannada → English, Tamil → Bengali, etc.