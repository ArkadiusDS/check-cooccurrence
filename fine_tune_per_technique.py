import json
import transformers
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

from utils.custom_callbacks import SaveMetricsCallback
from utils.utils import (
    PersuasionDataset, compute_metrics, predict_persuasion, compute_metrics_for_test_data
)
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split

LABELS = ['Appeal_to_Authority', 'Causal_Oversimplification',
          'Conversation_Killer', 'Exaggeration-Minimisation',
          'Guilt_by_Association', 'Flag_Waving', 'Appeal_to_Popularity',
          'Red_Herring', 'Name_Calling-Labeling', 'Repetition', 'Whataboutism',
          'Appeal_to_Fear-Prejudice', 'Doubt', 'Loaded_Language', 'Slogans',
          'Appeal_to_Hypocrisy', 'Straw_Man', 'False_Dilemma-No_Choice',
          'Obfuscation-Vagueness-Confusion']

transformers.set_seed(123)

train_data = pd.read_csv("train_persuasion_en.csv")
validation_data = pd.read_csv("dev_persuasion_en.csv")
validation_data, test_data = train_test_split(validation_data, test_size=0.30, random_state=42)

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-large-uncased")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased")

train_encodings = tokenizer(
    train_data['text'].tolist(),
    truncation=True,
    padding=True,
    max_length=512
)
val_encodings = tokenizer(
    validation_data['text'].tolist(),
    truncation=True,
    padding=True,
    max_length=512
)


for label in LABELS:

    train_dataset = PersuasionDataset(train_encodings, train_data[label].tolist())
    val_dataset = PersuasionDataset(val_encodings, validation_data[label].tolist())

    hyper_parameters_dict = {
        "evaluation_strategy": "steps",
        "learning_rate": 0.00001,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "num_train_epochs": 5,
        "warmup_steps": 200,
        "weight_decay": 0.03,
        "fp16": True,
        "metric_for_best_model": "f1_macro_weighted",
        "load_best_model_at_end": True,
        "save_total_limit": 2,
        "greater_is_better": True,
        "save_strategy": "steps",
        "eval_steps": 100
    }

    args = TrainingArguments(
        output_dir="output/new/" + label + "/",
        evaluation_strategy="steps",
        learning_rate=0.00001,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        warmup_steps=200,
        weight_decay=0.03,
        fp16=True,
        metric_for_best_model="f1_macro_weighted",
        load_best_model_at_end=True,
        save_total_limit=2,
        greater_is_better=True,
        save_strategy="steps",
        eval_steps=100,
        seed=123
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[SaveMetricsCallback(
            csv_file_name="metrics" + label + ".csv",
            hyperparameters=hyper_parameters_dict)
        ]
    )
    trainer.train()
    model_saved_path = "output/saved/" + label + "/"
    trainer.save_model(model_saved_path)

    print("######################## Unseen data evaluation ########################################")

    model = AutoModelForSequenceClassification.from_pretrained(model_saved_path)

    pred_column = "pred" +label
    test_data[pred_column] = test_data.text.apply(
                    lambda x: predict_persuasion(x, tokenizer, model)
                )

    evaluation_results = compute_metrics_for_test_data(test_data[label], test_data[pred_column])
    # Save evaluation metrics to a JSON file
    output_file_path = "output/test_metrics"+label+".json"

    with open(output_file_path, 'w') as output_file:
        json.dump(evaluation_results, output_file, indent=4)
