""" 
Funciones que limpian y normalizan los datos.
"""

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


def normalize_band(df, tpo_name, flux_name, error_name):
    df[tpo_name] = (df[tpo_name] - df[tpo_name].min()) / (df[tpo_name].max() - df[tpo_name].min())
    df[flux_name] = (df[flux_name] - df[flux_name].min()) / (df[flux_name].max() - df[flux_name].min())
    df[error_name] = (df[error_name] - df[error_name].min()) / (df[error_name].max() - df[error_name].min())

    return df

def cleaning_data(df_ligthcurves, n_det_min, tpo_name, name_col_flux, name_col_error, normalize=False):
    """Limpia los datos y normaliza los flujos""" 
    # data
    df_ligthcurves = df_ligthcurves.dropna() # Eliminación de datos NAN
    list_ids = df_ligthcurves.oid.unique() 

    # outuput
    lc_pos = []
    id_sn_clean = []

    for id_sn in list_ids:
        filter_lc = df_ligthcurves[df_ligthcurves.oid == id_sn]
        bands_name = filter_lc.fid.unique()
        bands_name = ['g', 'r'] # En este caso solo se esta utilizando estas dos bandas (eliminar si se generaliza)

        bands = []
        flag = True
        for band in bands_name:
            band_x = filter_lc[filter_lc.fid == band]

            # Para normalizar
            if normalize:
                band_x = normalize_band(band_x, tpo_name, name_col_flux, name_col_error)
            bands.append(band_x)

            # BBDD con puntos desde la explosión con y sin norm
                # with 'and' --> 1713 supernovas
                # 'or' --> 1926 supernovas (En este caso se utiliza 'or')
            if len(band_x) >= n_det_min and flag:
                id_sn_clean.append(id_sn)
                flag = False

        filter_lc_pos = pd.concat(bands, axis=0).sort_values(by=['mjd'])
        lc_pos.append(filter_lc_pos)

    df_lc_pos = pd.concat(lc_pos, axis=0)

    if normalize:
        df_lc_pos = df_lc_pos.dropna() # # Eliminación de datos NAN post normalization
        print(f"Numero de supernovas despues de la limpieza y normalización: {len(id_sn_clean)}")
    else:
        print(f"Numero de supernovas despues de la limpieza: {len(id_sn_clean)}")
        
    df_lc_pos_clean = df_lc_pos[df_lc_pos.oid.isin(id_sn_clean)]
    
    return df_lc_pos_clean