# -*- coding: UTF-8 -*-

import itertools
from math import factorial, log
import numpy as np
from numpy.lib.function_base import average
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from copy import deepcopy

DATA_DEFAULT_PATH = './datos/HT_Sensor_dataset.dat'
METADATA_DEFAULT_PATH = './datos/HT_Sensor_metadata.dat'
DATA_TRAIN_DEFAULT_PATH = './datos/data_train.csv'
DATA_TEST_DEFAULT_PATH = './datos/data_treal.csv'


def juntar_datos_metadatos(datos, metadatos):
    """ Recibe conjunto de datos y de metadatos y hace un natural join por id,
    devolviendo un dataframe con el resultado

    @param datos: conjunto de datos
    @type datos: dataframe pandas
    @param metadatos: conjunto de datos
    @type metadatos: dataframe pandas
    @rtype: dataframe
    """
    join = datos.set_index('id').join(metadatos.set_index('id'), how='inner')
    return join


# def crear_paquetes(datos_join, freq_paquete=30, unidades_por_freq=1):
#     """ Para cada muestra original (agrupando por id), la divide cada tiempo
#     determinado (freq_paquete, en segundos) y coloca todas las muestras
#     anteriores de los últimos freq_paquete * unidades_por_freq segundos. Por
#     ejemplo, si freq_paquete=30 y unidades_por_freq es 2, se simula que
#     cada 30 segundos se creara un paquete con las muestras de los últimos
#     60 segundos.

#     @param datos_join: Resultado inmediato tras unir datos y metadatos
#     @type datos_join: Dataframe
#     @param freq_paquete: Cada cuántos segundos se toma una medida
#     @type freq_paquete: int
#     @param unidades_por_freq: Cuántos múltiplos de la frecuencia se utilizan
#     para el tamaño final del paquete
#     @type unidades_por_freq: int
#     @rtype: lista de listas de diccionarios
#     """
#     agrupacion_id = datos_join.groupby('id')
#     ids = agrupacion_id.groups.keys()
#     paquetes = []
#     for id in ids:  # Para cada grupo original generar los paquetes
#         grupo_id = agrupacion_id.get_group(id)
#         paquete = []
#         t_inicial = None
#         for fila in grupo_id.to_dict('records'):
#             if t_inicial is None:
#                 t_inicial = fila['time']
#                 paquete.append(fila)
#             elif fila['time'] - t_inicial > freq_paquete / 3600:
#                 paquetes.append(paquete)
#                 t_inicial = fila['time']
#                 paquete = [fila]
#             else:
#                 paquete.append(fila)
#         if paquete:
#             paquetes.append(paquete)
#     # Añadir la cola de los paquetes
#     if unidades_por_freq > 1:
#         for i in range(len(paquetes) - 1, 0, -1):
#             for j in range(i - 1, max(i - unidades_por_freq, -1), - 1):
#                 paquetes[i] = deepcopy(paquetes[j]) + paquetes[i]
#     # Añadir un identificador a cada paquete
#     for id, paquete in enumerate(paquetes):
#         for fila in paquete:
#             fila['id'] = id
#     return paquetes


def crear_paquetes(datos_join, freq_paquete=30, unidades_por_freq=1):
    """ Para cada muestra original (agrupando por id), la divide cada tiempo
    determinado (freq_paquete, en segundos) y coloca todas las muestras
    anteriores de los últimos freq_paquete * unidades_por_freq segundos. Por
    ejemplo, si freq_paquete=30 y unidades_por_freq es 2, se simula que
    cada 30 segundos se creara un paquete con las muestras de los últimos
    60 segundos.

    @param datos_join: Resultado inmediato tras unir datos y metadatos
    @type datos_join: Dataframe
    @param freq_paquete: Cada cuántos segundos se toma una medida
    @type freq_paquete: int
    @param unidades_por_freq: Cuántos múltiplos de la frecuencia se utilizan
    para el tamaño final del paquete
    @type unidades_por_freq: int
    @rtype: lista de listas de diccionarios
    """
    freq_paquete /= 3600  # Conversion a horas
    agrupacion_id = datos_join.groupby('id')
    ids = agrupacion_id.groups.keys()
    paquetes = []
    for id in ids:  # Para cada grupo original generar los paquetes
        grupo_id = agrupacion_id.get_group(id)
        t_inicial = grupo_id['time'].min()
        t_final = grupo_id['time'].max()
        n_paquetes = int((t_final - t_inicial) / freq_paquete) + 1
        paquetes_nuevos = [list() for _ in range(n_paquetes)]
        for fila in grupo_id.to_dict('records'):
            pos = int((fila['time'] - t_inicial) / freq_paquete)
            paquetes_nuevos[pos].append(fila)
        # Añadir la cola de los paquetes
        if unidades_por_freq > 1:
            for i in range(len(paquetes_nuevos) - 1, 0, -1):
                for j in range(i - 1, max(i - unidades_por_freq, -1), - 1):
                    paquetes_nuevos[i] = (paquetes_nuevos[j] +
                                          paquetes_nuevos[i])
        paquetes += paquetes_nuevos
    # Añadir un identificador a cada paquete
    id = 0
    ids = []
    paquetes_f = []
    for paquete in paquetes:
        if paquete:
            paquetes_f.append(paquete)
            ids += [id] * len(paquete)
            id += 1
    return paquetes_f, ids


def empaquetar_muestras(datos_join, freq_paquete=30, unidades_por_freq=1):
    """
    Dado un conjunto de muestras las empaqueta según la frecuencia y unidiades
    por frecuencia pasados como argumento (véase crear_paquetes).
    Finalmente clasifica y etiqueta cada paquete según un voto por mayoría
    (el peso de cada muestra aumenta linealmente con el tiempo).

    @param datos_join: Resultado inmediato tras unir datos y metadatos
    @type datos_join: Dataframe
    @param freq_paquete: Cada cuántos segundos se toma una medida
    @type freq_paquete: int, optional
    @param unidades_por_freq: Cuántos múltiplos de la frecuencia se utilizan
    para el tamaño final del paquete
    @type unidades_por_freq: int, optional
    @rtype: Dataframe
    """
    paquetes, ids = crear_paquetes(datos_join, freq_paquete, unidades_por_freq)
    # Etiquetar cada paquete. La clase es voto por mayoría ponderado, donde
    # el peso de cada muestra del paquete aumenta cuadraticamente con el tiempo
    clases = []
    for paquete in paquetes:
        votacion = [0, 0]
        n = len(paquete)
        # Suma de t = 1 hasta n de log(t)
        denom_peso = n * (n + 1) // 2
        # denom_peso = log(factorial(n))
        denom_peso = n
        clase_actual = None
        for t, fila in enumerate(paquete):
            clase_actual = fila['clase']
            peso = (t + 1) / denom_peso
            if fila['time'] < 0 or fila['time'] > fila['dt']:
                votacion[0] += peso
            else:
                votacion[1] += peso
        clase = 'background' if votacion[0] > votacion[1] else clase_actual
        clases.append(clase)
    r = pd.DataFrame(list(itertools.chain(*paquetes)))
    r = r.drop(columns=['clase'])
    r['id'] = ids
    return r, clases


def filtrar_fuera_induccion(datos_join):
    """ Elimina la parte de las mediciones que no se corresponde con el momento
    en que se acerca el estímulo al sensor.

    @param datos_join: datos ya juntados
    @type datos_join: dataframe
    @rtype: dataframe
    """
    datos_join = datos_join[((datos_join.time > 0) &
                            (datos_join.time <= datos_join.dt)) |
                            (datos_join.clase == 'background')]
    return datos_join


def eliminar_metadata_sobrante(datos_join, atrs=['date', 't0', 'time', 'dt']):
    """ Elimina los metadatos sobrantes (una vez filtrados).

    @param datos_join: [description]
    @type datos_join: [type]
    @rtype: [type]
    @rtype: dataframe
    """
    return datos_join.drop(columns=atrs)


def leer_conjunto_datos(nombre_fichero):
    """ Lee el fichero de datos y devuelve dataframe de pandas

    @param nombre_fichero: nombre del fichero
    @type nombre_fichero: string
    @rtype: dataframe
    """
    return pd.read_csv(nombre_fichero, header=0, sep=r'\s+')


# https://cmdlinetips.com/2019/10/pandas-groupby-13-functions-to-aggregate/
def agrupar_datos(datos, agg_funcs=['count', 'sum', 'mean', 'median', 'min',
                                    'max', 'std', 'var', 'sem', 'describe']):
    """ Agrupa los datos por id usando las funciones de agregación indicadas

    @param datos: dataframe con los datos
    @type datos: dataframe
    @param agg_funcs: funciones de agregación, por defecto, las más comunes
    @type agg_funcs: list, optional
    @rtype: dataframe
    """
    return datos.groupby(['id', 'clase']).agg(agg_funcs).fillna(-1)


def transformacion_numpy(datos_group):
    """ Transformación a array de numpy del conjunto de datos ya agrupado por
    id y con su clase correspondiente

    @param datos_join: dataframe resultante de join + group
    @type datos_join: dataframe
    @rtype: tupla con el conjunto de datos y con el set de etiquetas
    """
    return (datos_group.to_numpy(),
            datos_group.index.get_level_values('clase').to_numpy())


# Busqueda de mejores parametros con grid search
def grid_search(X, y, test_size, scores, init_clf, tuned_parameters):
    """ Búsqueda de mejores hiperparámetros con Grid-Search
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html

    @param X: Datos totales
    @type X: Numpy array
    @param y: Clases
    @type y: Numpy array
    @param test_size: Tamaño de conjunto de test para cada permutación de los
    hiperparámetros
    @type test_size: float
    @param scores: forma de evaluar los hiperparámetros, vease
    https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    @type scores: list['string]
    @param init_clf: referencia al constructor del clasificador
    @type init_clf: función
    @param tuned_parameters: diccionario donde las claves son atributos del
    clasificador y los valores son lista de valores que tomará dicho atributo
    en cada permutación
    @type tuned_parameters: dicc {'string':list}
    """
    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=0)

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(
            init_clf(), tuned_parameters, scoring='%s_macro' % score,
            n_jobs=-1,  # Todos los procesadores
        )
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        # print()
        # print("Grid scores on development set:")
        # print()
        # means = clf.cv_results_['mean_test_score']
        # stds = clf.cv_results_['std_test_score']
        # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        #     print("%0.3f (+/-%0.03f) for %r"
        #           % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()


def crear_dataframe_original(funciones_agregacion=['min', 'max', 'sum', 'mean',
                                                'median', 'std', 'var', 'sem'],
                          output_dir=DATA_TRAIN_DEFAULT_PATH,
                          data_dir=DATA_DEFAULT_PATH,
                          metadata_dir=METADATA_DEFAULT_PATH):
    # Leer conjuntos de datos
    df_datos = leer_conjunto_datos(data_dir)
    df_metadatos = leer_conjunto_datos(metadata_dir)
    # Unir datos y metadatos
    df_join = juntar_datos_metadatos(df_datos, df_metadatos)
    # Quitar la porción de las muestras fuera de la inducción
    df_join = filtrar_fuera_induccion(df_join)
    # Quitar columnas que no aportan nada
    df_join = eliminar_metadata_sobrante(df_join)
    # Agrupar por id con las funciones de agregación indicadas
    df_res = agrupar_datos(df_join, agg_funcs=funciones_agregacion)
    df_res.reset_index(inplace=True)
    df_res.set_index('id', inplace=True)
    df_res.to_csv(output_dir)
    return df_res


def crear_dataframe_treal(funciones_agregacion=['min', 'max', 'sum', 'mean',
                                                'median', 'std', 'var', 'sem'],
                          freq_paquete=30,
                          unidades_por_freq=2,
                          output_dir=DATA_TEST_DEFAULT_PATH,
                          data_dir=DATA_DEFAULT_PATH,
                          metadata_dir=METADATA_DEFAULT_PATH):
    # Leer conjuntos de datos
    df_datos = leer_conjunto_datos(data_dir)
    df_metadatos = leer_conjunto_datos(metadata_dir)
    # Unir datos y metadatos
    df_join = juntar_datos_metadatos(df_datos, df_metadatos)
    # Empaquetar los datos para simular una situación más realista
    df_join, clases = empaquetar_muestras(df_join, freq_paquete,
                                          unidades_por_freq)
    # Quitar columnas que no aportan nada
    df_join = eliminar_metadata_sobrante(df_join)
    # Agrupar por id con las funciones de agregación indicadas
    df_res = df_join.groupby(['id']).agg(funciones_agregacion).fillna(0)
    df_res.insert(0, 'clase', clases)
    df_res.reset_index(inplace=True)
    df_res.set_index('id', inplace=True)
    df_res.to_csv(output_dir)
    return df_res


def csv_a_numpy(path):
    df = pd.read_csv(path, header=[0, 1, 2])
    arr = df.to_numpy()
    np.random.shuffle(arr)
    cols = list(df.columns[2:])
    for i in range(len(cols)):
        cols[i] = (str(cols[i][0]) + '_' + str(cols[i][1]))
    return arr[:, 2:].astype(np.float), arr[:, 1].astype(np.object), cols


if __name__ == '__main__':
    # crear_dataframe_train()
    crear_dataframe_treal(freq_paquete=60, unidades_por_freq=2)
    print(csv_a_numpy(DATA_TEST_DEFAULT_PATH))
