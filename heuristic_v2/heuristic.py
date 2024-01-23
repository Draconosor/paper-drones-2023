import pandas as pd
import numpy as np
from geopy.distance import geodesic
from typing import *

class Nodes:
    """
    Creates nodes with their respective attributes
    """
    def __init__(self, id: str, node_type: str, coordinates: Tuple[float, float], demand: int) -> None:
        self._id = id
        self._node_type = node_type
        self._coordinates = coordinates
        self._demand = demand
        self.isVisited = False

    def __repr__(self):
        return self.id
    
    @property
    def id(self) -> str:
        return self._id
    @property
    def node_type(self) -> str:
        return self._node_type
    @property
    def coordinates(self) -> Tuple[float, float]:
        return self._coordinates
    @property
    def X(self) -> float:
        return self._coordinates[0]
    @property
    def Y(self) -> float:
        return self._coordinates[1]
    
class Vehicle:
    """
    Creates a basic vehicle
    """
    def __init__(self, id: str, capacity: float, emissions: float) -> None:
        self._id = id
        self._capacity = capacity
        self._emissions = emissions
        self.route: List[Nodes] = []
        self.used_capacity = 0

    @property
    def id(self) -> str:
        return self._id
    @property
    def capacity(self) -> float:
        return self._capacity
    @property
    def emissions(self) -> float:
        return self._emissions
    @property
    def pct_used_capacity(self) -> float:
        return round(self.used_capacity*100/self.capacity,2)
    
class Drone(Vehicle):
    """
    Creates a drone and define its properties and attributes
    """
    def __init__(self, id: str, capacity: float, emissions: float, max_distance: float, weight: float) -> None:
        super().__init__(id, capacity, emissions)
        self._max_distance = max_distance
        self._weight = weight

    def __repr__(self) -> str:
        route_str = " -> ".join(str(node) for node in self.route)
        return f"Drone {self.id} with used capacity of: {self.pct_used_capacity}% with route {route_str}"
    
    @property
    def max_distance(self) -> float:
        return self._max_distance
    @property
    def weight(self) -> float:
        return self._weight
    
class Truck(Vehicle):
    """
    Creates a truck and define its properties and attributes
    """
    def __init__(self, id: str, capacity: float, emissions: float) -> None:
        super().__init__(id, capacity, emissions)
        self.drones: List[Drone] = []

    def __repr__(self) -> str:
        route_str = " -> ".join(str(node) for node in self.route)
        return f"Truck {self.id} with used capacity of: {self.pct_used_capacity}% with route {route_str} loaded with the following drones {self.drones}"

class Solution:
    """
    Stores a solution of the LRP-VRPD
    """
    def __init__(self, id: str,trucks: List[Truck]) -> None:
        self._id = id
        self._trucks = trucks

    @property
    def objective_values(self):
        ### AQUI TOCA DEFINIR COMO CALCULAR LOS OBJETIVOS (USAR VECTORES SI ES POSIBLE)###
        pass

    @property
    def id(self):
        return self._id
    
    def export(self):
        ### AQUI TOCA PONER LOS CRITERIOS PARA EXPORTAR UNA SOLUCION ###
        pass

def read_instance(instance_id: str) -> Union[List[Nodes], pd.DataFrame, pd.DataFrame, List[Truck], List[Drone]]:
    """Reads an instance file and creates all the relative objects associated with it

    Args:
        instance_id (str): The instance identifier (C#P#T#D#)

    Returns:
        Union[List[Nodes], pd.DataFrame, pd.DataFrame, List[Truck], List[Drone]]: The list of nodes, the distance matrices, the trucks and the drones defined from the instance parameters.
    """

    instance_file = f"instances/{instance_id}/{instance_id}.xlsx"
    nodes_df = pd.read_excel(instance_file, sheet_name="COORDS").merge(pd.read_excel(instance_file, sheet_name="DEMANDA"), left_on="NODES", right_on = "NODO")
    nodes_df["NODES"] = nodes_df["NODES"].astype("str")

    # CREATE NODES
    instance_nodes = [
        Nodes(
            row["NODES"],
            "Customer" if "Customer" in row["NODES"] else
            "Parking Lot" if "Depot" in row["NODES"] else
            "0",
            (row["X"], row["Y"]),
            row["DEMANDA"]
        )
        for _, row in nodes_df.iterrows()
    ]

    # READ PROBLEM SCALARS
    parameters = pd.read_excel(instance_file, sheet_name="PARAMETROS", usecols ="E:N").dropna().T.rename(columns= {0:"value"}).to_dict()["value"]
    
    # CREATE TRUCKS AND DRONES
    instance_trucks = [
        Truck(f"T{n}",
              parameters["QTR"],
              parameters["ETR"]
        )
        for n in range(1,int(parameters["NTRUCKS"])+1)
    ]
    instance_drones = [
        Drone(f"D{n}",
              parameters["QDR"],
              parameters["EDR"],
              parameters["MAXDDR"],
              parameters["WDR"]

        )
        for n in range(1,int(parameters["NDRONES"])+1)
    ]

    # READ MATRICES

    googlemaps_dm = pd.read_excel(instance_file, sheet_name="MANHATTAN", index_col=0)
    
    haversine_dm = pd.read_excel(instance_file, sheet_name="EUCLI", index_col=0)

    return instance_nodes, googlemaps_dm, haversine_dm, instance_trucks, instance_drones

a,b,c,d,e = read_instance('C10P10T5D10')