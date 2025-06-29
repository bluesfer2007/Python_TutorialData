from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

# --- CONFIGURACIÓN OFFLINE ---
# Ruta a la carpeta local donde guardaste el modelo de NLP Town.
ruta_modelo_local = "./modelo_bert_descargado" 

print("Cargando el modelo de calificación por estrellas desde archivos locales...")

# Cargar el tokenizador y el modelo desde la carpeta local
try:
    tokenizer = AutoTokenizer.from_pretrained(ruta_modelo_local)
    model = AutoModelForSequenceClassification.from_pretrained(ruta_modelo_local)
    print("¡Modelo cargado exitosamente en modo offline!")
except OSError:
    print(f"Error: No se encontraron los archivos del modelo en la ruta '{ruta_modelo_local}'.")
    print("Por favor, asegúrate de haber ejecutado primero el script 'descargar_modelo_bert.py'.")
    exit()


# --- FUNCIÓN DE PREDICCIÓN ADAPTADA ---
def predict_star_rating(textos):
    """
    Analiza una lista de textos y devuelve su calificación en estrellas (1 a 5).
    """
    inputs = tokenizer(textos, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    # La salida del modelo (logits) se convierte a predicciones.
    # El modelo produce una clase de 0 a 4.
    predicciones = torch.argmax(outputs.logits, dim=-1).tolist()

    # IMPORTANTE: Mapear la salida del modelo a estrellas.
    # El modelo fue entrenado para que 'label 0' sea 1 estrella, 'label 1' sea 2 estrellas, etc.
    mapa_estrellas = {
        0: "★☆☆☆☆ (1 estrella)",
        1: "★★☆☆☆ (2 estrellas)",
        2: "★★★☆☆ (3 estrellas)",
        3: "★★★★☆ (4 estrellas)",
        4: "★★★★★ (5 estrellas)"
    }

    return [mapa_estrellas[p] for p in predicciones]

# --- EJEMPLO DE USO ---
# ¡Ahora puedes desconectar tu internet para probar!

# Lista de reseñas de productos para analizar
textos_de_prueba = [
    # Español
    "El producto es una maravilla, superó todas mis expectativas.",
    "No está mal, pero podría ser mejor. Cumple su función.",
    "Una completa decepción, el material es de pésima calidad y no funciona.",

    # Inglés
    "This is the best purchase I have ever made!",
    "It's an okay product, nothing special.",

    # Francés
    "Je suis très déçu par cet article, il s'est cassé après deux jours.",

    # Alemán
    "Absolut fantastisch! Sehr zu empfehlen."
]

df= pd.read_csv("../test_comentarios.csv",sep=";")
textos_de_prueba = df['Texto'].tolist()  # Cargar los textos desde

# Obtener las predicciones
calificaciones = predict_star_rating(textos_de_prueba)

# Imprimir los resultados
print("\n--- Resultados del Análisis de Calificación por Estrellas (Offline) ---")
for texto, calificacion in zip(textos_de_prueba, calificaciones):
    print(f"Reseña: {texto}\nCalificación: {calificacion}\n")