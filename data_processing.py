import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def import_data(data_path='data'):
    # Definir la ruta del directorio y listar los archivos
    data_path = 'data'
    files = os.listdir(data_path)

    # Lista para guardar los DataFrames
    dfs = []

    # Crear figura para los plots
    plt.figure(figsize=(12, 6))

    # Leer y plotear cada archivo
    for file in files:
        if file.startswith('data') and file.endswith('.csv'):
            file_path = os.path.join(data_path, file)
            df = pd.read_csv(file_path)

            # Guardar el DataFrame en la lista
            dfs.append(df)

    # Concatenar todos los DataFrames
    df_total = pd.concat(dfs, ignore_index=True)

    df_total['FECHA'] = pd.to_datetime(df_total['FECHA'])

    return df_total


def missing_data_interpolation(df_total, n):

    df = df_total[df_total['NRO_CAJERO'] == n].sort_values('FECHA')

    df = df.copy()
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    df = df.sort_values('FECHA').reset_index(drop=True)

    nro_cajero = df['NRO_CAJERO'].iloc[0]

    # Creamos un diccionario de acceso rápido por fecha
    datos = {row['FECHA']: row for _, row in df.iterrows()}

    # Fechas completas esperadas
    fechas_completas = pd.date_range(df['FECHA'].min(), df['FECHA'].max(), freq='D')

    # Vamos construyendo el nuevo conjunto de datos
    nuevas_filas = {}

    for i in reversed(range(len(fechas_completas) - 1)):
        fecha = fechas_completas[i]
        fecha_sig = fechas_completas[i + 1]

        if fecha not in datos:
            # Buscar en datos + nuevas_filas la fila siguiente
            if fecha_sig in nuevas_filas:
                fila_siguiente = nuevas_filas[fecha_sig]
            elif fecha_sig in datos:
                fila_siguiente = datos[fecha_sig]
            else:
                continue  # No tenemos datos para calcular

            nueva_fila = {
                'FECHA': fecha,
                'NRO_CAJERO': nro_cajero,
                'REMANENTE': fila_siguiente['REMANENTE'] + fila_siguiente['DISPENSADO'],
                'DISPENSADO': 0
            }

            nuevas_filas[fecha] = nueva_fila

    # Combinar original con nuevas filas
    df_final = pd.concat([
        df,
        pd.DataFrame(nuevas_filas.values())
    ]).sort_values('FECHA').reset_index(drop=True)

    return df_final

# Se hace en el data_loader.py
# def split_date(df_resultado):
#     df_resultado['AÑO'] = df_resultado['FECHA'].dt.year
#     df_resultado['MES'] = df_resultado['FECHA'].dt.month
#     df_resultado['DIA'] = df_resultado['FECHA'].dt.day
#     df_resultado['WEEKDAY'] = df_resultado['FECHA'].dt.weekday

#     return df_resultado


def add_inflation(df_resultado, inflacion_path='data/inflation.csv'):
    # Leer el archivo de inflación
    df_inflacion = pd.read_csv(inflacion_path)

    # Crear un diccionario para acceder rápidamente a la inflación por año y mes
    inflacion_dict = {(row['año'], row['mes']): row['inflacion'] for _, row in df_inflacion.iterrows()}

    # Función para obtener la inflación del mes anterior
    def obtener_inflacion_anterior(fecha):
        año = fecha.year
        mes = fecha.month
        
        if mes == 1:  # Si es enero, tomar diciembre del año anterior
            mes_anterior = 12
            año_anterior = año - 1
        else:  # Para otros meses, tomar el mes anterior del mismo año
            mes_anterior = mes - 1
            año_anterior = año
        
        return inflacion_dict.get((año_anterior, mes_anterior), None)

    # Aplicar la función para calcular la inflación del mes anterior
    df_resultado['INFLACION_MES_ANTERIOR'] = df_resultado['FECHA'].apply(obtener_inflacion_anterior)

    return df_resultado


def add_different_day_means(df_resultado):
    df_resultado['MEDIA_3_DIAS'] = df_resultado['DISPENSADO'].rolling(window=3, min_periods=1).mean()
    df_resultado['MEDIA_7_DIAS'] = df_resultado['DISPENSADO'].rolling(window=7, min_periods=1).mean()
    df_resultado['MEDIA_15_DIAS'] = df_resultado['DISPENSADO'].rolling(window=15, min_periods=1).mean()
    df_resultado['MEDIA_30_DIAS'] = df_resultado['DISPENSADO'].rolling(window=30, min_periods=1).mean()

    return df_resultado


def add_important_days(df_resultado, dias_importantes_path='data/dias_importantes.csv'):
    # Leer el archivo de días importantes
    df_dias_importantes = pd.read_csv(dias_importantes_path)

    # Crear una columna de fecha en el DataFrame de días importantes
    df_dias_importantes['FECHA'] = pd.to_datetime(
        dict(year=df_dias_importantes['año'], 
            month=df_dias_importantes['mes'], 
            day=df_dias_importantes['día'])
    )

    # Crear un conjunto de fechas importantes para rápida comparación
    fechas_importantes = set(df_dias_importantes['FECHA'])

    # Agregar columna al df_resultado indicando si el día es importante
    df_resultado['ES_DIA_IMPORTANTE'] = df_resultado['FECHA'].apply(lambda x: 1 if x in fechas_importantes else 0)

    return df_resultado


def final_details(df_resultado):
    df_resultado['TARGET'] = df_resultado['DISPENSADO'].rolling(window=7, min_periods=1).sum()
    df_resultado = df_resultado.drop(columns=['NRO_CAJERO'])
    
    return df_resultado

def process_data(n):
    
    os.makedirs('data_processed', exist_ok=True)
    df_total = import_data('data') 

    k = min(n, df_total['NRO_CAJERO'].max())

    for i in range(0, k+1):
        df_resultado = missing_data_interpolation(df_total, i)
        #df_resultado = split_date(df_resultado)
        df_resultado = add_inflation(df_resultado, 'data/inflation.csv')
        df_resultado = add_different_day_means(df_resultado)
        df_resultado = add_important_days(df_resultado, 'data/dias_importantes.csv')
        df_resultado = final_details(df_resultado)

        df_resultado.to_csv(f'data_processed/df_resultado_cajero_{i}.csv', index=False)

    print("Todos los archivos fueron guardados en la carpeta 'data_processed'.")

