from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
import torch
import os

# 1. Parameter
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HF_TOKEN = "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"
DATA_PATH = "participant_prompts.jsonl"
OUTPUT_DIR = "/nethome/hhelbig/Neural_Networks/LLM_tuning/llama_test"
LOG_DIR = "/nethome/hhelbig/Neural_Networks/LLM_tuning/llama3-test-finetuned"
MAX_LENGTH = 2048

# 2. Tokenizer laden
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, use_auth_token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# 3. Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# 4. Modell laden
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    use_auth_token=HF_TOKEN,
    torch_dtype=torch.float16,
)

# 5. LoRA anwenden
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# 6. Dataset laden & splitten
dataset = load_dataset("json", data_files=DATA_PATH)["train"]
dataset = dataset.train_test_split(test_size=0.1, seed=42)

def format_example(example):
    prompt = example["prompt"].strip()
    completion = example["completion"].strip()
    full_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}\n<|start_header_id|>assistant<|end_header_id|>\n{completion}<|eot_id|>"
    return {"text": full_text}

train_dataset = dataset["train"].map(format_example)
eval_dataset = dataset["test"].map(format_example)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

train_dataset = train_dataset.map(tokenize, remove_columns=["text"])
eval_dataset = eval_dataset.map(tokenize, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 7. Training arguments mit Early Stopping (via Callback)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=20,  # mehr Epochs
    save_strategy="epoch",
    eval_strategy="epoch",  # evaliere jedes Epoch-Ende
    save_total_limit=2,
    logging_dir=LOG_DIR,
    logging_steps=10,
    logging_first_step=True,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
    report_to="tensorboard",
    load_best_model_at_end=True,  # bestes Modell nach Early Stopping laden
    metric_for_best_model="eval_loss",  # welche Metrik nutzen
    greater_is_better=False,
)

from transformers import EarlyStoppingCallback

class ExampleGenerationCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        # Ein paar fixe Prompts zum Testen:
        test_prompts = [
            "Eine Person hat folgende Eigenschaften und hat folgende Angaben gemacht. Auf Basis dieser Informationen: Wird diese Person ihre Krankenversicherung in den nächsten 6 Monaten wechseln? Bitte antworte mit \"Nicht-Wechsler ohne Wechselgedanken l10y\", \"Nicht-Wechsler mit Wechselgedanken l10y\" oder \"Wechsler l10y\".\n\nSoziodemografie:\nDie Person ist Weiblich, ist 34 Jahre alt, lebt in Berlin, In der gesetzlichen Krankenversicherung versichert, verdient 66.600 € bis 92.999 €.\n\nHaushalt & Beruf:\nHaushaltsgröße: Mehr als 3 Personen, hat 2 Kinder, Migrationshintergrund: Ja, aus: Osteuropa, arbeitet: Vollzeit, Berufserfahrung: Seit weniger als 5 Jahren, Beziehungsstatus: In einer Partnerschaft.\n\nMarkenbekanntheit: Die Person kann folgende Krankenkassen ohne Hinweis nennen:\nTk, Bkk, Barmer, Aok, Die Person kennt folgende Krankenkassen, wenn sie genannt werden: AOK, IKK Brandenburg und Berlin, DAK Gesundheit, Techniker Krankenkasse (TK), KKH Kaufmännische Krankenkasse, BAHN-BKK, Barmer.\n\nNutzung & Wechsel:\nist bei der Techniker Krankenkasse (TK) versichert, hat über einen Wechsel Noch nie nachgedacht.\n\nWissen / Selbstwahrnehmung:\nschätzt ihr Wissen über GKV als Wenig bis gar nicht ein.\n\nWechselgründe – allgemein & konkret: Die Person gibt folgendes als Wechselgründe an:\nÄnderung meiner Lebenssituation (z.B. Familienzuwachs, Krankheit, Arbeitgeberwechsel, Ausbildung etc.), Schlechte Erfahrungen mit meiner Krankenkasse (z.B. Leistungsablehnung, schlechter Service etc.), Kosten/ Beitrag meiner Krankenkasse, Die Person antwortet auf folgende Frage: Bitte wählen Sie eine Situation, welche Sie am wahrscheinlichsten dazu veranlassen würde, Ihre gesetzliche Krankenkasse wechseln zu wollen., mit: Schlechte Erfahrungen mit meiner Krankenkasse (z.B. Leistungsablehnung, schlechter Service etc.), Die Person antwortet auf folgende Frage: Bitte geben Sie genauer an, welche Situation Sie am wahrscheinlichsten dazu veranlassen würde, über einen Krankenkassenwechsel nachzudenken., mit: Unfreundlicher Kundenservice.\n\nZufriedenheit / Leistungsbewertung:\nDie Person bewertet die Aussage: Sehr gutes Angebot eines Bonusprogramms z.B. Zusätzliche Gesundheits-Check-Ups, Auszahlungen, Sportprogramme etc als 'Trifft eher zu' bei der eigenen Krankenkasse, bewertet die Aussage :Hervorragendes Angebot zusätzlicher Leistungen (z.B. Zahnreinigung, Osteopathie, Kurse und Programme zur Gesundheitsvorsorge wie Rückenschule, Raucherentwöhnung oder Gewichtsreduktion, etc als 'Weder noch' bei der eigenen Krankenkasse, bewertet die Aussage: Größtes Angebot an weiteren Zusatzversicherungen z.B. Zahnzusatz, Auslandskrankenschutz etc als 'Weder noch' bei der eigenen Krankenkasse, bewertet die Aussage:Tarife mit Beitragsrückerstattungen z.B. Wahltarif oder Gutscheinen für Zusatzleistungen bei gesundheitsbewusstem Verhalten als 'Trifft eher zu' bei der eigenen Krankenkasse, bewertet die Aussage: Besondere Versorgungsangebote z.B. besondere Versorgung für spezifische Erkrankungen, Arztterminvermittlung, Zweitmeinung, telemedizinische Versorgung, etc. als 'Weder noch' bei der eigenen Krankenkasse, bewertet die Aussage: Vorreiter in Sachen Innovation bspw. Telemedizin als 'Weder noch' bei der eigenen Krankenkasse, bewertet die Aussage: Exzellente Beratung zu verschiedenen Gesundheitsthemen (z.B. zu Leistungen, Vorsorgeuntersuchungen, Krankheiten und Präventionsmaßnahmen als 'Trifft eher zu' bei der eigenen Krankenkasse, bewertet die Aussage: Umfangreicher digitaler Kundenservice z.B. Live-/Videochat, Apps, Chatbot, papierlose Dokumentenübermittlung, etc. als 'Trifft eher zu' bei der eigenen Krankenkasse, bewertet die Aussage: Uneingeschränkte Erreichbarkeit des Kundenservice z.B. Öffnungszeiten, Geschäftsstelle vor Ort als 'Trifft eher zu' bei der eigenen Krankenkasse, bewertet die Aussage: Äußerst vertrauensvoller und unterstützender Kundenumgang als 'Trifft vollkommen zu' bei der eigenen Krankenkasse, bewertet die Aussage: Schnellstmögliche Bearbeitung von Anliegen als 'Trifft eher zu' bei der eigenen Krankenkasse, bewertet die Aussage: Niedriger Preis (Geringer Zusatzbeitrag im Vergleich zu anderen Krankenkassen) als 'Trifft eher zu' beid der eigenen Krankenkasse, bewertet die Aussage: Hervorragender Ruf der Krankenkasse als 'Trifft vollkommen zu' bei der eigenen Krankenkasse, bewertet die Aussage: Nachweisliches Engagement der Krankenkasse zum Thema Nachhaltigkeit als 'Weder noch' bei der eigenen Krankenkasse.\n\nLoyalität / Empfehlung:\nDie Person fühlt sich ihrer GKV Eher verbunden , auf einer Skala von 1 bis 10 würde ihre Krankenkasse mit '10/10' weiterempfehlen, Begründung für Weiterempfehlung: \"Immer zufrieden gewesen\".\n\nInformationsverhalten & Abschluss:\nDie Person informiert sich über Freunde & Familie, Ärzte/ Apotheken/ med. Fachpersonal, die ausschlaggebende Informationsquelle ist: Freunde & Familie.\n\nWechselbarrieren:\nDie Person fühlt sich auf einer Skala von 1 bis 13 'Rank 1/13' verbunden mit der Krankenkasse, Die Person bewertet auf einer Skala von 1 bis 13 die Angst vor schlechteren Leistungen nach einem Wechsel als 'Rank 2/13', auf einer Skala von 1 bis 13, bewertet die Person die Aussage: Mich hat keine andere Krankenkasse überzeugt als 'Rank 3/13'.\n\nPreisakzeptanz:\nhält einen Wechsel bei einem Zusatzbeitrag für Eher unwahrscheinlich.\n\nKontaktverhalten:\nOnline Kontakt mit GKV: Nie, telefonischer Kontakt mit GKV: ca. 7-12-mal im Jahr, Kontakt in Geschäftsstelle: Nie, Briefkontakt mit GKV: ca. 3 bis 6-mal im Jahr, bevorzugter Kontaktweg mit der GKV: Telefonisch, bevorzugter Kontaktaufnahme von der GKV: Per Brief, letzter Kontakt war: Vor weniger als einem Monat, kontaktierte die GKV wegen Arbeitsunfähigkeitszeiten/-bescheinigungen, bewertet den Kontakt mit der GKV als Sehr positiv, positives Kontaktbeispiel: Freundlicher & empathischer Kundenservice.\n\nZusatzversicherungen & weitere Anbieter:\nDie Person hat folgende Zusatzversicherungen: \"1\".\n\nAndere Verträge::\nStrom-/Gasanbieter: 1-mal gewechselt, Mobilfunkanbieter: 2 bis 3-mal gewechselt, Internetanbieter: 1-mal gewechselt, Kfz-Versicherung: Noch nie gewechselt.\n\nPKV Wechsel:\nGedanken zum Wechsel in die PKV: Nein, ich habe mich mit einem Wechsel nicht beschäftigt, Wechselgründe: Höhere Kosten im Alter, Keine kostenlose Mitversicherung von Kindern, Ich befürworte das Solidarprinzip der gesetzlichen Krankenkasse, Wechselinteresse zur PKV: Äußerst unwahrscheinlich.\n\nGesundheit & Prävention:\nDie Person trifft folgende Gesundheitsvorsorge: Ich gehe regelmäßig zu Vorsorgeuntersuchungen, Ich treibe mehrmals die Woche intensiv Sport, Ich bin sehr gesundheitsbewusst, hat folgende Erkrankungen: Bandscheibenvorfall und/oder chronische Rückenschmerzen, weitere Erkrankungen: In keinem DMP eingeschrieben / möchte die Frage nicht beantworten."
        ]
        model.eval()
        print(f"\n--- Beispielgenerierungen nach Epoche {state.epoch:.0f} ---")
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
            gen_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            print(f"Prompt: {prompt}\nAntwort: {gen_text}\n")
        model.train()

# 8. Trainer mit Callbacks
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3), ExampleGenerationCallback()]
)

# 9. Training starten
trainer.train()
print("Training abgeschlossen")

# 10. Modell speichern
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
print(f"✅ Fine-tuning complete. Model saved to {OUTPUT_DIR}")
