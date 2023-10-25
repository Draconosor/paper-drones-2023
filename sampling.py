import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from typing import *
from haversine import haversine


def manhattan_distance(point1 : Tuple[float], point2: Tuple[float]) -> float:
    """
    Calculate the Manhattan distance between two points on the Earth's surface.
    
    :param point1: Tuple with latitude and longitude of the first point in degrees.
    :param point2: Tuple with latitude and longitude of the second point in degrees.
    
    :return: The Manhattan distance between the two points in kilometers.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = np.radians(point1)
    lat2, lon2 = np.radians(point2)

    # Calculate the absolute differences in latitude and longitude
    delta_lat = abs(lat2 - lat1)
    delta_lon = abs(lon2 - lon1)

    # Convert the differences to kilometers (assuming Earth's radius is 6371 km)
    lat_distance = delta_lat * 6371
    lon_distance = delta_lon * 6371

    # Calculate the Manhattan distance
    distance = lat_distance + lon_distance

    return distance

def sample_generator(sample_size: int, parking_size: int, n_trucks:int, n_drones:int, avg_demand: int = 350, std_demand: int = 50):
    np.random.seed(0)
    data = pd.read_excel("Data Sampleo.xlsx", sheet_name = 'parking_coords')
    x_mean = data['Longitud'].mean()
    y_mean = data['Latitud'].mean()
    x_std = data['Longitud'].std()
    y_std = data['Latitud'].std()
    sample_data = {'Customer' : [f'Cliente{x}' for x in range(1,sample_size+1)],
                'Longitud' : np.random.normal(x_mean, x_std, sample_size), 
                'Latitud' : np.random.normal(y_mean, y_std, sample_size)}
    sample_df = pd.DataFrame(sample_data)
    sampled_parkings = data.loc[np.random.choice(data.index, parking_size, replace=False)].copy().reset_index(drop = True)
    node_0 = pd.DataFrame(data = [['0', sampled_parkings['Latitud'].mean(), sampled_parkings['Longitud'].mean()]], columns=sampled_parkings.columns)

    sample_coords = pd.concat([node_0,sampled_parkings, sample_df.rename(columns={'Customer': 'Depot'})]).rename(columns={'Depot': 'NODES', 'Longitud':'X',  'Latitud':'Y'})\
        .reset_index(drop = True)

    sample_parameters = pd.concat([sample_coords.NODES, 
                                   pd.Series(data = range(1,n_trucks+1), name='TRUCKS'), 
                                   pd.Series(data = range(1,n_drones+1), name='DRONES'),
                                   pd.Series(name=''),
                                   pd.DataFrame({"NDEPOTS":{"0":5},"NCUSTOMERS":{"0":10},"NTRUCKS":{"0":4},"NDRONES":{"0":5},"ETR":{"0":410},"EDR":{"0":41},"QTR":{"0":5000},"QDR":{"0":420},"MAXDDR":{"0":10},"WDR":{"0":360}}).reset_index(drop=True)
                                   ], axis=1)
    sample_demand = pd.DataFrame({
                                  "NODO": sample_coords.NODES,
                                  "DEMANDA" : [0 for x in range(0,parking_size+1)] + np.ceil(np.random.normal(avg_demand, std_demand, sample_size)).tolist()     
                                    })
    
    coordinates = sample_coords.set_index('NODES')[['Y', 'X']].to_numpy()
    manhattan_dm = pd.DataFrame(cdist(coordinates, coordinates, metric=manhattan_distance), index=sample_coords['NODES'], columns=sample_coords['NODES']).replace(0,100000)
    euclidean_dm = pd.DataFrame(cdist(coordinates, coordinates, metric=haversine), index=sample_coords['NODES'], columns=sample_coords['NODES']).replace(0,100000)

    with pd.ExcelWriter(f'instances/C{sample_size}P{parking_size}T{n_trucks}D{n_drones}.xlsx', engine='xlsxwriter') as writer:
        sample_parameters.to_excel(writer, sheet_name='PARAMETROS', index = False)
        sample_demand.to_excel(writer, sheet_name='DEMANDA', index = False)
        sample_coords.set_index('NODES').to_excel(writer, sheet_name='COORDS')
        manhattan_dm.to_excel(writer, sheet_name='MANHATTAN')
        euclidean_dm.to_excel(writer, sheet_name='EUCLI')

sample_generator(50,10,5,10)