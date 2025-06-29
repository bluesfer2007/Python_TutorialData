import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# --- MAPAS DE RESULTADOS PARA CADA MODELO ---
# Mapa para el modelo Tabularis (5 etiquetas)
MAPA_TABULARIS = {0: "Muy Negativo", 1: "Negativo", 2: "Neutral", 3: "Positivo", 4: "Muy Positivo"}

# Mapa para el modelo NLP Town Bert (5 estrellas)
MAPA_NLP_BERT = {
    0: "Muy Negativo (1 estrella)",
    1: "Negativo (2 estrellas)",
    2: "Neutral (3 estrellas)",
    3: "Positivo (4 estrellas)",
    4: "Muy Positivo (5 estrellas)"
}

def cargar_modelo_local(ruta_modelo, device):
    """
    Carga un tokenizador y un modelo desde una ruta local y lo mueve al dispositivo especificado (CPU o GPU).
    """
    if not os.path.exists(ruta_modelo):
        print(f"¡ERROR! La ruta del modelo '{ruta_modelo}' no existe.")
        print("Asegúrate de que la estructura de carpetas coincida con la imagen y que los modelos estén descargados.")
        return None, None
    
    try:
        print(f"Cargando modelo desde: {ruta_modelo}")
        tokenizer = AutoTokenizer.from_pretrained(ruta_modelo)
        model = AutoModelForSequenceClassification.from_pretrained(ruta_modelo)
        model.to(device) # Mover el modelo a la GPU si está disponible
        model.eval() # Poner el modelo en modo de evaluación
        print("...Carga exitosa.")
        return tokenizer, model
    except Exception as e:
        print(f"Ocurrió un error al cargar el modelo desde '{ruta_modelo}': {e}")
        return None, None

def analizar_lote(textos, tokenizer, model, mapa_resultados, device):
    """
    Función genérica para analizar un lote de textos con un modelo dado.
    """
    if not textos:
        return []
        
    # Tokenizar el lote de textos
    inputs = tokenizer(textos, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Mover los tensores de entrada al mismo dispositivo que el modelo
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Realizar la predicción sin calcular gradientes (más rápido)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Obtener la predicción (el índice con la probabilidad más alta)
    predicciones = torch.argmax(outputs.logits, dim=-1).tolist()
    
    # Mapear los índices a las etiquetas de texto correspondientes
    return [mapa_resultados[p] for p in predicciones]


def procesar_archivo_completo(config):
    """
    Función principal que orquesta todo el proceso de análisis.
    """
    # 1. Verificar si la GPU está disponible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Usando dispositivo: {device} ---")
    if device == "cuda":
        print(f"Nombre de la GPU: {torch.cuda.get_device_name(0)}")

    # 2. Cargar los dos modelos desde las rutas locales
    print("\n--- Cargando Modelos Locales ---")
    tokenizer_tabularis, model_tabularis = cargar_modelo_local(config["ruta_tabularis"], device)
    tokenizer_nlp_bert, model_nlp_bert = cargar_modelo_local(config["ruta_nlp_bert"], device)

    # Salir si alguno de los modelos no pudo cargarse
    if not model_tabularis or not model_nlp_bert:
        print("\nProceso detenido debido a un error en la carga de modelos.")
        return

    # 3. Cargar el archivo de Excel
    try:
        print(f"\n--- Cargando datos de '{config['archivo_entrada']}' ---")
        df = pd.read_excel(config['archivo_entrada'])
        print(f"Archivo cargado. {len(df)} filas encontradas.")
    except FileNotFoundError:
        print(f"¡ERROR! No se encontró el archivo '{config['archivo_entrada']}'. Verifica el nombre y la ubicación.")
        return
    except Exception as e:
        print(f"Ocurrió un error al leer el archivo Excel: {e}")
        return

    # Extraer los comentarios y asegurarse de que no haya valores nulos
    comentarios = df[config['columna_texto']].fillna('').tolist()
    
    # Limitar filas si se especificó un límite para pruebas
    if config['limite_filas']:
        print(f"\n¡ATENCIÓN! Se procesarán únicamente las primeras {config['limite_filas']} filas para la prueba.")
        comentarios = comentarios[:config['limite_filas']]
        df = df.head(config['limite_filas']).copy()

    # 4. Procesar en lotes
    print(f"\n--- Iniciando análisis de {len(comentarios)} comentarios en lotes de {config['tamano_lote']} ---")
    resultados_tabularis = []
    resultados_nlp_bert = []

    for i in range(0, len(comentarios), config['tamano_lote']):
        lote_textos = comentarios[i:i + config['tamano_lote']]
        
        # Analizar con el primer modelo
        res_tab = analizar_lote(lote_textos, tokenizer_tabularis, model_tabularis, MAPA_TABULARIS, device)
        resultados_tabularis.extend(res_tab)
        
        # Analizar con el segundo modelo
        res_bert = analizar_lote(lote_textos, tokenizer_nlp_bert, model_nlp_bert, MAPA_NLP_BERT, device)
        resultados_nlp_bert.extend(res_bert)

        print(f"Procesados {len(resultados_tabularis)} de {len(comentarios)} comentarios...")

    # 5. Añadir resultados al DataFrame
    df['sentimiento_tabularis'] = resultados_tabularis
    df['calificacion_nlp_bert'] = resultados_nlp_bert
    print("\n--- Análisis completado. Añadiendo resultados al archivo. ---")

    # 6. Guardar el archivo final
    try:
        df.to_csv(config['archivo_salida'], index=False, encoding='utf-8-sig')
        print(f"\n¡ÉXITO! Resultados guardados en '{config['archivo_salida']}'")
    except Exception as e:
        print(f"Ocurrió un error al guardar el archivo de salida: {e}")


# --- CONFIGURACIÓN Y EJECUCIÓN ---
if __name__ == "__main__":
    
    # Diccionario de configuración para mantener el código limpio
    configuracion = {
        # 1. Archivos y columnas
        "archivo_entrada": "Data_Clasif_comentarios.xlsx",
        "columna_texto": "Comentario", # Nombre de la columna a analizar
        "archivo_salida": "resultados_analisis_combinado.csv",

        # 2. Rutas a tus modelos locales (basado en tu imagen)
        "ruta_tabularis": "local_tabuilaris_sentiment/modelo_descargado",
        "ruta_nlp_bert": "local_nlp_bert/modelo_bert_descargado",

        # 3. Parámetros de ejecución
        "tamano_lote": 100, # Reduce si tienes errores de memoria en la GPU, aumenta si tienes mucha VRAM
        "limite_filas": None # Pon un número pequeño para pruebas rápidas. Para procesar todo, pon: None
    }
    
    procesar_archivo_completo(configuracion)