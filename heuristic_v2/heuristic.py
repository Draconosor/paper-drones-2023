import pandas as pd
import os
from typing import List, Tuple

from tables import Node


class Nodes:
    """Creates nodes with their respective attributes."""
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
    """Base class for vehicles."""
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
        return round(self.used_capacity * 100 / self.capacity, 2)


class Drone(Vehicle):
    """Creates a drone with specific attributes."""
    def __init__(self, id: str, capacity: float, emissions: float, max_distance: float, weight: float) -> None:
        super().__init__(id, capacity, emissions)
        self._max_distance = max_distance
        self._weight = weight

    def __repr__(self) -> str:
        route_str = " -> ".join(str(node) for node in self.route)
        return f"Drone {self.id} ({self.pct_used_capacity}% capacity) with route: {route_str}"

    @property
    def max_distance(self) -> float:
        return self._max_distance

    @property
    def weight(self) -> float:
        return self._weight


class Truck(Vehicle):
    """Creates a truck with specific attributes."""
    def __init__(self, id: str, capacity: float, emissions: float) -> None:
        super().__init__(id, capacity, emissions)
        self.drones: List[Drone] = []

    def __repr__(self) -> str:
        route_str = " -> ".join(str(node) for node in self.route)
        return f"Truck {self.id} ({self.pct_used_capacity}% capacity) with route: {route_str} and drones: {self.drones}"


def read_instance_file(instance_id: str) -> pd.ExcelFile:
    """Reads the instance Excel file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    file_path = os.path.join(project_root, "instances", instance_id, f"{instance_id}.xlsx")
    return pd.ExcelFile(file_path)


def create_nodes(instance_file: pd.ExcelFile) -> List[Nodes]:
    """Creates nodes from the instance data."""
    coords_df = pd.read_excel(instance_file, sheet_name="COORDS")
    demand_df = pd.read_excel(instance_file, sheet_name="DEMANDA")
    nodes_df = coords_df.merge(demand_df, left_on="NODES", right_on="NODO")
    nodes_df["NODES"] = nodes_df["NODES"].astype(str)

    return [
        Nodes(
            row["NODES"],
            "Customer" if "Customer" in row["NODES"] else "Parking Lot" if "Depot" in row["NODES"] else "0",
            (row["X"], row["Y"]),
            row["DEMANDA"]
        )
        for _, row in nodes_df.iterrows()
    ]


def create_vehicles(instance_file: pd.ExcelFile) -> Tuple[List[Truck], List[Drone]]:
    """Creates trucks and drones from the instance parameters."""
    parameters = pd.read_excel(instance_file, sheet_name="PARAMETROS", usecols="E:N").dropna().T
    params = parameters.rename(columns={0: "value"}).to_dict()["value"]

    trucks = [Truck(f"T{n}", params["QTR"], params["ETR"]) for n in range(1, int(params["NTRUCKS"]) + 1)]
    drones = [
        Drone(f"D{n}", params["QDR"], params["EDR"], params["MAXDDR"], params["WDR"])
        for n in range(1, int(params["NDRONES"]) + 1)
    ]
    return trucks, drones


def read_distance_matrices(instance_file: pd.ExcelFile) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reads distance and time matrices from the instance file."""
    def index_col_to_string(df: pd.DataFrame) -> pd.DataFrame:
        """Convert both index and columns of a DataFrame to string type."""
        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)
        return df 
    googlemaps_dm = index_col_to_string(pd.read_excel(instance_file, sheet_name="MANHATTAN", index_col=0))
    haversine_dm = index_col_to_string(pd.read_excel(instance_file, sheet_name="EUCLI", index_col=0))
    times_truck = index_col_to_string(pd.read_excel(instance_file, sheet_name="TIEMPOS_CAM", index_col=0))
    times_drone = index_col_to_string(pd.read_excel(instance_file, sheet_name="TIEMPOS_DRON", index_col=0))
    return googlemaps_dm, haversine_dm, times_truck, times_drone


def read_instance(instance_id: str) -> Tuple[List[Nodes], pd.DataFrame, pd.DataFrame, List[Truck], List[Drone], pd.DataFrame, pd.DataFrame]:
    """Reads an instance file and creates associated objects."""
    instance_file = read_instance_file(instance_id)
    instance_nodes = create_nodes(instance_file)
    instance_trucks, instance_drones = create_vehicles(instance_file)
    googlemaps_dm, haversine_dm, times_truck, times_drone = read_distance_matrices(instance_file)

    return instance_nodes, googlemaps_dm, haversine_dm, instance_trucks, instance_drones, times_truck, times_drone


# Example usage:
instance_nodes, googlemaps_dm, haversine_dm, instance_trucks, instance_drones, times_truck, times_drone = read_instance('C10P5T5D15')


def nn_bigroute(base_nodes: List[Nodes], distance_matrix: pd.DataFrame):
    nodes = [node for node in base_nodes if node._node_type != 'Parking Lot']
    ncustomers = len([node for node in nodes if node._node_type == 'Customer'])
    route = []
    neighbours = {}
    distance_matrix.at['0','0']
    #construccion de vecinos ordenados de menor a mayor
    for i in nodes:
        vecinos = 0
        rawlist = []
        while vecinos < len(nodes) -1:
            distmin = 99999
            candidate = nodes[0]
            for j in nodes:
                if i.id != j.id and distance_matrix.at[i.id,j.id] < distmin and j not in rawlist:
                    distmin = distance_matrix.at[i.id,j.id]
                    candidate = j
            rawlist.append(candidate)
            vecinos += 1
        neighbours[i.id] = rawlist
    #construccion de ruta
    i = nodes[0]
    route.append(i)
    visited = 0
    while visited < ncustomers:
        for j in neighbours[i.id]:
            if j not in route:
                route.append(j)
                i = j
                visited += 1
                break
    route.append(nodes[0])
    return route

route = nn_bigroute(instance_nodes, googlemaps_dm)