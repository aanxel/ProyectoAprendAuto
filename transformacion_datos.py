import numpy as np
import pandas as pd


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
    return datos_group.to_numpy(), datos_group.index.get_level_values('class')


# def entropia(datos_group, atributo, agregación):
#     datos_atributo = df_res.reset_index('id')[(atributo, agregación)]
#     g_sum = datos_atributo[atributo].transform('sum')
#     values = datos_[atributo]]/g_sum
#     df[str(atributo)+'_entropia'] = -(values*np.log(values))

#     df1 = df.groupby('Name_Receive',as_index=False,sort=False)['Entropy'].sum()


if __name__ == '__main__':
    df_datos = leer_conjunto_datos(DATA_DEFAULT_PATH)
    df_metadatos = leer_conjunto_datos(METADATA_DEFAULT_PATH)
    df_join = juntar_datos_metadatos(df_datos, df_metadatos)
    df_join = eliminar_metadata_sobrante(filtrar_fuera_induccion(df_join))
    df_res = agrupar_datos(df_join, agg_funcs=['min', 'max'])
    X, y = transformacion_numpy(df_res)
    print(df_res.reset_index('id')[('R1', 'min')])
    df_res.to_csv('./datos/transformacion.csv')