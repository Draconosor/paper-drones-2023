from copy import deepcopy
import enum
from time import time
import pandas as pd
import os
from typing import List, Tuple, Dict, Set
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

## Set Seed
np.random.seed(0)

class Nodes:
    """Creates nodes with their respective attributes."""
    def __init__(self, id: str, node_type: str, coordinates: Tuple[float, float], demand: int) -> None:
        self._id = id
        self._node_type = node_type
        self._coordinates = coordinates
        self._demand: float = demand
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
    
    def __eq__(self, other):
        if isinstance(other, Nodes):
            return self.id == other.id
        return False
    
    def __hash__(self):
        return hash(self.id)

class Vehicle:
    """Base class for vehicles."""
    def __init__(self, id: str, capacity: float, emissions: float) -> None:
        self._id = id
        self._capacity = capacity
        self._emissions = emissions
        self.route: List[Nodes] = []
        self.used_capacity: float = 0

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

    @property
    def is_used(self) -> bool:
        return self.used_capacity > 0
    
    def __eq__(self, other):
        if isinstance(other, Vehicle):
            return self.id == other.id
        return False
    
    def __hash__(self):
        return hash(self.id)

    def reset_vehicle(self):
        self.route = []

class Drone(Vehicle):
    """Creates a drone with specific attributes."""
    def __init__(self, id: str, capacity: float, emissions: float, max_distance: float, weight: float) -> None:
        super().__init__(id, capacity, emissions)
        self._max_distance = max_distance
        self._weight: float = weight
        self.assigned_to: str = ''
        self.visit_node: Nodes = None

    def __repr__(self) -> str:
        route_str = " -> ".join(str(node) for node in self.route)
        return f"Drone {self.id} ({self.pct_used_capacity}% capacity) {f'with route: {route_str} assigned to {self.assigned_to}' if not self.route == [] else ''}"
    
    def reset_vehicle(self):
        super().reset_vehicle()
        self.assigned_to = ''
        self.visit_node = None

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
    
    def reset_vehicle(self):
        super().reset_vehicle()
        self.drones = []

@dataclass
class RouteCache:
    emissions: Dict[tuple, float] = None
    
    def __post_init__(self):
        self.emissions = {}

def calculate_route_metric(route: List[Nodes], matrix: pd.DataFrame, cache: RouteCache) -> float:
    """Cached calculation of route metrics (emissions or time)"""
    route_key = tuple(node.id for node in route)
    if route_key not in cache.emissions:
        cache.emissions[route_key] = sum(matrix.at[route[i].id, route[i+1].id] for i in range(len(route)-1))
    return cache.emissions[route_key]

def nn_bigroute(base_nodes: List[Nodes], distance_matrix: pd.DataFrame) -> List[Nodes]:
    """Optimized Nearest-Neighbor algorithm for Big Route generation"""
    nodes = [node for node in base_nodes if node._node_type != 'Parking Lot']
    ncustomers = len([node for node in nodes if node._node_type == 'Customer'])
    
    # Precompute distances
    distances = {(i.id, j.id): distance_matrix.at[i.id, j.id] 
                for i in nodes for j in nodes if i != j}
    
    # Initialize route
    route = [nodes[0]]
    visited = {nodes[0]}
    
    while len(visited) < ncustomers + 1:
        current = route[-1]
        next_node = min(
            (node for node in nodes if node not in visited),
            key=lambda x: distances.get((current.id, x.id), float('inf'))
        )
        route.append(next_node)
        visited.add(next_node)
    
    route.append(nodes[0])
    return route

class Saving:
    def __init__(self, parking_lot: Nodes, to: Nodes, saving: float, prev_node: Nodes = None, truck_a: Truck = None, truck_b: Truck = None) -> None:
        self.parking_lot = parking_lot
        self.to = to
        self.prev_node = prev_node
        self._saving = saving
        self.truck_a = truck_a
        self.truck_b = truck_b
        self.used = False
        
    @property
    def saving(self):
        return self._saving
    
    @property
    def strict_nodes(self):
        return [self.parking_lot, self.to]
        
    def __repr__(self) -> str:
        return f'Saving from {self.prev_node} to parking {self.parking_lot.id} using drone to {self.to.id} of value {self.saving}'
    
    def __eq__(self, other):
        if isinstance(other, Saving):
            return self.saving == other.saving
        return False
    
    def __lt__(self, other):
        if not isinstance(other, Saving):
            return NotImplemented
        return self.saving < other.saving

    def __gt__(self, other):
        if not isinstance(other, Saving):
            return NotImplemented
        return self.saving > other.saving

def assign_drones_bigroute(base_nodes: List[Nodes], bigroute: List[Nodes], 
                          drones: List[Drone], truck_matrix: pd.DataFrame, 
                          drone_matrix: pd.DataFrame) -> None:
    """Optimized drone assignment to bigroute"""
    cache = RouteCache()
    parking_lots = {node for node in base_nodes if node.node_type == 'Parking Lot'}
    
    # Precompute route indices and initial emissions
    node_indices = {node: idx for idx, node in enumerate(bigroute)}
    initial_emissions = calculate_route_metric(bigroute, truck_matrix, cache)
    
    # Calculate all valid savings
    savings = []
    for i, current in enumerate(bigroute[1:-1]):
        if current.node_type != 'Customer':
            continue
            
        for p_lot in parking_lots:
            for j, target in enumerate(bigroute[i+1:-1], i+1):
                if target.node_type != 'Customer' or target._demand > drones[0].capacity:
                    continue
                    
                new_route = bigroute.copy()
                new_route[j] = p_lot
                
                new_emissions = calculate_route_metric(new_route, truck_matrix, cache)
                drone_emissions = 2 * drone_matrix.at[p_lot.id, target.id]
                
                saving = initial_emissions - (new_emissions + drone_emissions)
                if saving > 0:
                    savings.append(Saving(p_lot, target, saving))
    
    # Sort savings once
    savings.sort(reverse=True)
    
    # Assign drones efficiently
    used_nodes = set()
    for drone in drones:
        for saving in savings:
            if (not saving.used and saving.to not in used_nodes and 
                saving.to._demand <= drone.capacity and 
                2 * drone_matrix.at[saving.parking_lot.id, saving.to.id] <= drone.max_distance):
                
                fly_to = saving.to
                p_lot = saving.parking_lot
                
                # Update drone
                drone.visit_node = fly_to
                drone.route = [p_lot, fly_to, p_lot]
                drone.used_capacity = fly_to._demand
    
                if p_lot._demand > 0:
                    bigroute.pop(bigroute.index(fly_to))
                else:
                    bigroute[bigroute.index(fly_to)] = p_lot
                
                # Update parking lot and route
                p_lot._demand += fly_to._demand
                saving.used = True
                used_nodes.add(fly_to)
                break

def cluster_bigroute(bigroute: List[Nodes], trucks: List[Truck], base_nodes: List[Nodes], drones: List[Drone]) -> None:
    """Optimized clustering of bigroute into truck routes"""
    depot = base_nodes[0]
    route_wt_depot = bigroute[1:-1]
    
    # Track assignments
    assignments = defaultdict(list)
    drone_assignments = defaultdict(list)
    
    for node in route_wt_depot:
        assigned = False
        for truck in trucks:
            if not node.isVisited and truck.used_capacity + node._demand <= truck.capacity:
                if node.node_type == 'Parking Lot':
                    relevant_drones = [d for d in drones if node in d.route]
                    if all(truck.used_capacity + d.weight + d.visit_node._demand <= truck.capacity 
                          for d in relevant_drones):
                        for drone in relevant_drones:
                            drone.assigned_to = truck.id
                            drone_assignments[truck].append(drone)
                            truck.used_capacity += drone.weight + drone.visit_node._demand
                
                assignments[truck].append(node)
                truck.used_capacity += node._demand
                node.isVisited = True
                assigned = True
                break
        
        if not assigned:
            # Handle unassigned nodes
            for truck in trucks:
                if not truck.is_used:
                    assignments[truck].append(node)
                    truck.used_capacity += node._demand
                    node.isVisited = True
                    break
    
    # Build final routes
    for truck in trucks:
        if assignments[truck]:
            truck.route = [depot] + assignments[truck] + [depot]
            truck.drones = drone_assignments[truck]

def single_2opt_improvement(route: List[Nodes], distance_matrix: pd.DataFrame, cache: RouteCache) -> List[Nodes]:
    """
    Performs a single 2-opt improvement on the route if possible.
    Returns the improved route or the original route if no improvement is found.
    """
    current_distance = calculate_route_metric(route, distance_matrix, cache)
    
    for i in range(1, len(route) - 2):
        for j in range(i + 1, len(route) - 1):
            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
            new_distance = calculate_route_metric(new_route, distance_matrix, cache)
            
            if new_distance < current_distance:
                return new_route
                
    return route

def twoopt_until_local_optimum(route: List[Nodes], distance_matrix: pd.DataFrame, cache: RouteCache) -> List[Nodes]:
    """
    Applies 2-opt improvements repeatedly until no further improvements can be made.
    """
    improved = True
    while improved:
        new_route = single_2opt_improvement(route, distance_matrix, cache)
        improved = new_route != route
        route = new_route
            
    return route


def drone_launch(base_nodes: List[Nodes], drones: List[Drone], trucks: List[Truck],
                 drone_matrix: pd.DataFrame, truck_matrix: pd.DataFrame) -> None:
    """Optimized drone launch implementation"""
    cache = RouteCache()
    parking_lots = {node for node in base_nodes if node.node_type == 'Parking Lot'}
    
    # Calculate savings per truck
    savings_per_truck = defaultdict(list)
    for truck in trucks:
        if len(truck.route) <= 3:
            continue
            
        route_emissions = calculate_route_metric(truck.route, truck_matrix, cache)
        
        for i, current in enumerate(truck.route[1:-1]):
            if current.node_type != 'Customer':
                continue
                
            for p_lot in parking_lots:
                for j, target in enumerate(truck.route[i+1:-1], i+1):
                    if target.node_type != 'Customer' or current == target:
                        continue
                        
                    new_route = truck.route.copy()
                    new_route[j] = p_lot
                    
                    new_emissions = calculate_route_metric(new_route, truck_matrix, cache)
                    drone_emissions = 2 * drone_matrix.at[p_lot.id, target.id]
                    
                    saving = route_emissions - (new_emissions + drone_emissions)
                    if saving > 0:
                        savings_per_truck[truck].append(
                            Saving(p_lot, target, saving, current)
                        )
        
        if savings_per_truck[truck]:
            savings_per_truck[truck].sort(reverse=True)
    
    # Assign drones
    used_drones = set()
    used_nodes = set()
    
    for truck in trucks:
        if truck not in savings_per_truck:
            continue
            
        for saving in savings_per_truck[truck]:
            available_drone = next(
                (d for d in drones if d not in used_drones 
                 and saving.to._demand <= d.capacity
                 and truck.used_capacity + d.weight <= truck.capacity
                 and 2 * drone_matrix.at[saving.parking_lot.id, saving.to.id] <= d.max_distance
                 and saving.to not in used_nodes),
                None
            )
            
            if available_drone:
                # Apply drone assignment
                p_lot = saving.parking_lot
                to_node = saving.to
                
                available_drone.visit_node = to_node
                available_drone.route = [p_lot, to_node, p_lot]
                available_drone.assigned_to = truck.id
                available_drone.used_capacity = to_node._demand
                
                truck.drones.append(available_drone)
                if p_lot in truck.route:
                    truck.route.remove(to_node)
                else:
                    truck.route[truck.route.index(to_node)] = p_lot
                    
                p_lot._demand += to_node._demand
                truck.used_capacity += available_drone.weight
                used_drones.add(available_drone)
                used_nodes.add(to_node)

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
    
    # Optimize merge operation
    nodes_df = coords_df.merge(demand_df, left_on="NODES", right_on="NODO", how='left')
    nodes_df["NODES"] = nodes_df["NODES"].astype(str)
    
    # Use list comprehension instead of apply
    return [
        Nodes(
            str(row["NODES"]),
            "Customer" if "Customer" in row["NODES"] 
            else "Parking Lot" if "Depot" in row["NODES"] 
            else "0",
            (row["X"], row["Y"]),
            row["DEMANDA"]
        )
        for _, row in nodes_df.iterrows()
    ]

def create_vehicles(instance_file: pd.ExcelFile) -> Tuple[List[Truck], List[Drone], float, float]:
    """Creates trucks and drones from the instance parameters."""
    # Optimize DataFrame operations
    parameters = pd.read_excel(
        instance_file, 
        sheet_name="PARAMETROS", 
        usecols="E:N"
    ).dropna().T
    
    params = parameters.rename(columns={0: "value"}).to_dict()["value"]
    
    # Create vehicles using list comprehensions
    trucks = [
        Truck(f"T{n}", params["QTR"], params["ETR"]) 
        for n in range(1, int(params["NTRUCKS"]) + 1)
    ]
    
    drones = [
        Drone(
            f"D{n}", 
            params["QDR"], 
            params["EDR"], 
            params["MAXDDR"]*params["EDR"], 
            params["WDR"]
        )
        for n in range(1, int(params["NDRONES"]) + 1)
    ]
    
    return trucks, drones, params["ETR"], params["EDR"]

def read_distance_matrices(instance_file: pd.ExcelFile, etr: float, edr: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Optimized reading of distance and time matrices."""
    def process_matrix(sheet_name: str, multiplier: float = 1.0) -> pd.DataFrame:
        df = pd.read_excel(instance_file, sheet_name=sheet_name, index_col=0)
        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)
        if multiplier != 1.0:
            df *= multiplier
        return df
    
    # Process all matrices at once
    googlemaps_dm = process_matrix("MANHATTAN", etr)
    haversine_dm = process_matrix("EUCLI", edr)
    times_truck = process_matrix("TIEMPOS_CAM")
    times_drone = process_matrix("TIEMPOS_DRON")
    
    return googlemaps_dm, haversine_dm, times_truck, times_drone

def read_instance(instance_id: str) -> Tuple[List[Nodes], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Truck], List[Drone]]:
    """Optimized instance reading function."""
    instance_file = read_instance_file(instance_id)
    
    # Parallel processing could be implemented here for larger instances
    instance_nodes = create_nodes(instance_file)
    instance_trucks, instance_drones, etr, edr = create_vehicles(instance_file)
    googlemaps_dm, haversine_dm, times_truck, times_drone = read_distance_matrices(instance_file, etr, edr)
    
    return instance_nodes, googlemaps_dm, haversine_dm, times_truck, times_drone, instance_trucks, instance_drones

def random_truck_paired_select(trucks: List[Truck]) -> List[Truck]:
    used_trucks = [truck for truck in trucks if truck.is_used and len(truck.route) >= 3]
    indices = np.arange(len(used_trucks))
    permuted_indices = np.random.permutation(indices)
    return [used_trucks[i] for i in permuted_indices]

def swap_interruta_random(truck_a: Truck, truck_b: Truck, matrix: pd.DataFrame, cache: RouteCache):
    failures = 0
    success = False
    while not success and failures <= 50:
        node_a: Nodes = np.random.choice(truck_a.route[1:-1])
        node_b: Nodes = np.random.choice(truck_b.route[1:-1])
        if truck_a.used_capacity - node_a._demand + node_b._demand <= truck_a.capacity and truck_b.used_capacity - node_b._demand + node_a._demand <= truck_b.capacity:
            print(node_a, node_b)
            truck_a.route[truck_a.route.index(node_a)] = node_b
            truck_a.used_capacity = truck_a.used_capacity - node_a._demand + node_b._demand
            truck_b.route[truck_b.route.index(node_b)] = node_a
            truck_b.used_capacity = truck_b.used_capacity - node_b._demand + node_a._demand
            truck_a.route = twoopt_until_local_optimum(truck_a.route, matrix, cache)
            truck_b.route = twoopt_until_local_optimum(truck_b.route, matrix, cache)
            success = True
        else:
            failures += 1

def open_truck(trucks: List[Truck], matrix: pd.DataFrame, cache: RouteCache):
    for i, truck in enumerate(trucks):
        if not truck.is_used and i > 0 and len(trucks[i-1].route) > 3:
            node_0 = trucks[i-1].route[0]
            prev_truck_route = trucks[i-1].route[1:-1]
            # Find the middle index
            middle_index = len(prev_truck_route) // 2

            # Split the list into two halves
            first_half = prev_truck_route[:middle_index]
            second_half = prev_truck_route[middle_index:]
            
            # Assign Routes
            ## Previous Truck
            trucks[i-1].route = first_half
            trucks[i-1].route.append(node_0)
            trucks[i-1].route.insert(0, node_0)
            
            ## New Truck
            for drone in trucks[i-1].drones:
                if drone.is_used:
                    if drone.route[0] in second_half:
                        drone.assigned_to = truck.id
                        truck.drones.append(drone)
                        
            trucks[i-1].drones = [d for d in trucks[i-1].drones if d.assigned_to == trucks[i-1].id]
            
            truck.route = second_half
            truck.route.append(node_0)
            truck.route.insert(0, node_0)
            
            twoopt_until_local_optimum(trucks[i-1].route, matrix, cache)
            twoopt_until_local_optimum(truck.route, matrix, cache)
            
            break
            
def close_parking_lot_random(trucks: List[Truck], truck_matrix: pd.DataFrame, drone_matrix: pd.DataFrame, truck_cache: RouteCache):
    truck = np.random.choice([t for t in trucks if t.is_used])
    lots = [p for p in truck.route if p.node_type == 'Parking Lot']
    if lots:
        to_close: Nodes = np.random.choice(lots)
        remaining = lots[:]
        remaining.remove(to_close)
        print(f'Closed {to_close} on truck {truck.id}')
        if remaining:
            for drone in truck.drones:
                if drone.route[0] == to_close:
                    ## Look for best node available
                    best_cost = 10e6
                    best_route = []
                    for p_lot in remaining:
                        route = [p_lot, drone.visit_node, p_lot]
                        cost = calculate_route_metric(route, drone_matrix, RouteCache())
                        if cost <= drone.max_distance and cost < best_cost:
                            best_cost = cost
                            best_route = route
                    if best_route:
                        drone.route = best_route
                    else:
                        truck.route.insert(-2, drone.visit_node)
                        drone.reset_vehicle()
                        
        twoopt_until_local_optimum(truck.route, truck_matrix, truck_cache)

          

def main(instance_id: str):
    """Main function with performance monitoring."""
    start_time = time()
    
    # Read instance
    instance_nodes, googlemaps_dm, haversine_dm, times_truck, times_drone, instance_trucks, instance_drones = read_instance(instance_id)
    
    # Create route cache for the entire solution process
    cache = RouteCache()
    
    # Generate initial solution
    bigroute = nn_bigroute(instance_nodes, googlemaps_dm)
    
    # Assign drones to bigroute
    assign_drones_bigroute(instance_nodes, bigroute, instance_drones, googlemaps_dm, haversine_dm)
    
    # Cluster bigroute into truck routes
    cluster_bigroute(bigroute, instance_trucks, instance_nodes, instance_drones)
    
    # Launch additional drones
    drone_launch(instance_nodes, instance_drones, instance_trucks, haversine_dm, googlemaps_dm)
    
    # Optimize all truck routes
    for truck in instance_trucks:
        if truck.route:
            truck.route = twoopt_until_local_optimum(truck.route, googlemaps_dm, cache)
    
    # Calculate final objective
    total_emissions = sum(
        calculate_route_metric(truck.route, googlemaps_dm, cache) +
        sum(calculate_route_metric(drone.route, haversine_dm, cache) 
            for drone in truck.drones)
        for truck in instance_trucks
    )
    end_time = time()
    execution_time = end_time - start_time
    
    return total_emissions, execution_time

if __name__ == "__main__":
    instance_id = 'C30P5T5D15'
    obj, runtime = main(instance_id)
    print(f"Final objective: {obj:.2f}")
    print(f"Execution time: {runtime:.2f} seconds")