import numpy as np
import pandas as pd
import itertools as itr
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC


DATA_DEFAULT_PATH = './datos/HT_Sensor_dataset.dat'
METADATA_DEFAULT_PATH = './datos/HT_Sensor_metadata.dat'


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


def filtrar_fuera_induccion(datos_join):
    """ Elimina la parte de las mediciones que no se corresponde con el momento
    en que se acerca el estímulo al sensor.

    @param datos_join: datos ya juntados
    @type datos_join: dataframe
    @rtype: dataframe
    """
    datos_join = datos_join[datos_join.time > 0]
    datos_join = datos_join[datos_join.time <= datos_join.dt]
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
    return datos.groupby(['id', 'class']).agg(agg_funcs)


def transformacion_numpy(datos_group):
    """ Transformación a array de numpy del conjunto de datos ya agrupado por
    id y con su clase correspondiente

    @param datos_join: dataframe resultante de join + group
    @type datos_join: dataframe
    @rtype: tupla con el conjunto de datos y con el set de etiquetas
    """
    return (datos_group.to_numpy(),
            datos_group.index.get_level_values('class').to_numpy())


# def _permutaciones_dicc_parametros(params):
#     lista_valores = []
#     lista_claves = []
#     for atr, valores in params.items():
#         lista_valores.append(valores)
#         lista_claves.append(atr)
#     permutaciones = itr.product(*lista_valores)
#     for permutacion_atrs in permutaciones:
#         yield {lista_claves[i]: permutacion_atrs[i]
#                for i in range(len(lista_claves))}


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


if __name__ == '__main__':
    # df_datos = leer_conjunto_datos(DATA_DEFAULT_PATH)
    # df_metadatos = leer_conjunto_datos(METADATA_DEFAULT_PATH)
    # df_join = juntar_datos_metadatos(df_datos, df_metadatos)
    # df_join = eliminar_metadata_sobrante(filtrar_fuera_induccion(df_join))
    # df_res = agrupar_datos(df_join, agg_funcs=['min', 'max'])
    # X, y = transformacion_numpy(df_res)
    # print(df_res.reset_index('id')[('R1', 'min')])
    # df_res.to_csv('./datos/transformacion.csv')
    test(SVC)
