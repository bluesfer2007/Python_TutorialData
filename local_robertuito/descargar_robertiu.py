# descargar_robertuito.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "pysentimiento/robertuito-sentiment-analysis"
SAVE_PATH = "./local_robertuito/modelo_robertuito_descargado"

print(f"📥 Descargando el modelo '{MODEL_ID}'...")

# Descargar y guardar el tokenizador y el modelo
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

print(f"✅ ¡Modelo Robertuito guardado exitosamente en '{SAVE_PATH}'!")