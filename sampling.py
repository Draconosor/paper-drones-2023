import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from typing import *
from haversine import haversine
import os
import gams
import base64
from time import sleep

def dedent(my_string:str) -> str:
    lines = my_string.splitlines()
    dedented_lines = [line.lstrip() for line in lines]
    dedented_string = "\n".join(dedented_lines)
    return dedented_string

def sample_generator(objective: str,sample_size: int, parking_size: int, n_trucks:int, n_drones:int, avg_demand: int = 4, std_demand: int = 2):

    model = f"""sets
                    i nodos
                    k Camiones
                    l Drones
                    alias(i,j)
                    ;
                    parameters
                    NDepots Numero de parqueaderos
                    NCustomers Numero de clientes
                    NDrones Numero de Drones
                    NTrucks Nuemro de Vehiculos
                    ETR Emision de los camiones
                    EDR Emision de los drones
                    QTR Capacidad de los camiones
                    QDR Capacidad de los drones
                    MAXDDR Autonomia del dron en distancia
                    WDR Peso del dron
                    DISTmh(i,j) MaTDist Manhattan
                    DISTec(i,j) MaTDist Euclidiana
                    TIMET(i,j) MatTemp Camiones
                    TIMED(i,j) MatTemp Drones
                    DEM(i) Demanda del nodo i
                    $GDXIN in.gdx
                    $LOAD i, k, l
                    $LOADDC NDepots, NCustomers, NDrones, NTrucks, ETR, EDR, QTR, QDR, MAXDDR, WDR, DISTmh, DISTec, TIMET, TIMED, DEM
                    $GDXIN
                    variables
                    x(i,j,k) 1 si el camion k va del nodo i al nodo j
                    y(i,j,k,l) 1 si el dron l que viaja en el camion k va del nodo i al nodo j
                    v(k) 1 si se usa el camion k
                    s(l) 1 si se usa el dron l
                    u(i) variable para subrutas
                    z1 Emisiones Totales
                    z2 Makespan del vehiculo k
                    binary variable x, y, v, s
                    integer variable u
                    free variables z1, z2
                    ;
                    equations
                    salirdepot(k) un vehiculo solo puede salir una vez del deposito
                    llegardepot(k) si el vehiculo sale del deposito debe llegar
                    llegadacliente(j) A cada cliente se llega una sola vez
                    flujored(j,k) si un vehiculo llega a un nodo debe salir de el
                    relcamiondron(i,k,l) relacionar uso de dron con uso de camion
                    relcamiondron2(j,k,l)
                    arranquedron(i,j,k,l) un dron solo puede salir de un parqueadero
                    capcamion(k) no se puede exceder la capacidad del camion
                    capcamion2(k)
                    capdron(l) no se puede exceder la capacidad del dron
                    capdron2(l)
                    capvuelodron(l,i) no se puede exceder la capacidad de vuelo del dron
                    subtours(i,j,k) se deben evitar sub rutas en el recorrido
                    dronporcamion(i,j,l) un dron puede ir en un camion
                    fo1 Calculo z1
                    epsi(k) Balanceo Tiempos
                    ;

                    salirdepot(k).. sum((j)$(ord(j) > 1), x('0',j,k)) =L= v(k);
                    llegardepot(k).. sum((i)$(ord(i) > 1), x(i,'0',k)) =E= sum((j)$(ord(j) > 1), x('0',j,k));
                    llegadacliente(j)$(ord(j) > (1+NDepots)).. sum((i,k)$(ord(i) <> ord(j)), x(i,j,k)) + sum((i,k,l)$(ord(i) <> ord(j)), y(i,j,k,l)) =E= 1;
                    flujored(j,k)$(ord(j) > 1).. sum(i$(ord(i) <> ord(j)), x(i,j,k)) =E= sum(i$(ord(i) <> ord(j)), x(j,i,k));
                    relcamiondron(i,k,l)$(ord(i) > 1 and ord(i) <= (1+NDepots)).. sum(j$(ord(i) <> ord(j)), y(i,j,k,l)) =L= sum(j$(ord(i) <> ord(j)), x(j,i,k));
                    relcamiondron2(j,k,l)$(ord(j) > 1 and ord(j) <= (1+NDepots)).. sum(i$(ord(j) <> ord(i)), y(i,j,k,l)) =L= sum(i$(ord(j) <> ord(i)), x(j,i,k));
                    arranquedron(i,j,k,l)$(ord(i) = 1 or ord(i) > (1+NDepots)).. y(i,j,k,l) =L= 0;
                    capcamion(k).. sum((i,j)$(ord(i) <> ord(j)), DEM(j) * x(i,j,k)) + sum((i,j,l)$(ord(i) <> ord(j)), (WDR+DEM(j))* y(i,j,k,l)) =L= QTR * v(k);
                    capcamion2(k).. sum((i,j)$(ord(i) <> ord(j)), DEM(j) * x(i,j,k)) + sum((i,j,l)$(ord(i) <> ord(j)), (WDR+DEM(j)) * y(i,j,k,l)) =G= v(k);
                    capdron(l).. sum((i,j,k)$(ord(i) <> ord(j)), DEM(j) * y(i,j,k,l)) =L= QDR * s(l);
                    capdron2(l).. sum((i,j,k)$(ord(i) <> ord(j)), DEM(j) * y(i,j,k,l)) =G= s(l);
                    capvuelodron(l,i).. sum((j,k)$(ord(i) <> ord(j)), DISTec(i,j) * y(i,j,k,l)) =L= MAXDDR;
                    subtours(i,j,k)$(ord(i) > 1 and ord(j) > 1).. u(i) - u(j) + card(i) * x(i,j,k) =L= card(i) - 1;
                    dronporcamion(i,j,l)$(ord(i) <> ord(j)).. sum((k),y(i,j,k,l)) =l= 1;
                    fo1.. z1 =E= ETR * sum((i,j,k)$(ord(i) <> ord(j)), DISTmh(i,j) * x(i,j,k)) + 2*EDR * sum((i,j,k,l)$(ord(i) <> ord(j)), DISTec(i,j) * y(i,j,k,l));
                    epsi(k).. z2(k) =E= sum((i,j), x(i,j,k)*TIMET(i,j)) + sum((i,j,l), y(i,j,k,l)*TIMED(i,j));

                    Model Modelo1 /all/;
                    option MIP=CPLEX
                    option optcr=0.00000000000001;
                    modelo1.Reslim = 7200;
                    set workmem 128;
                    set mip strategy file 2;
                    set mip limits treememory 10000;
                    Solve Modelo1 using mip minimizing {objective};
                    Display x.L, y.L, v.L, s.L, z1.L,z2.L;
                    """



    np.random.seed(0)
    cols = ['node', 'latitude','longitude']
    node_0 = pd.read_excel("Data Sampleo.xlsx", sheet_name = 'node_0')[cols]
    parkings = pd.read_excel("Data Sampleo.xlsx", sheet_name = 'parking_coords')
    customers = pd.read_excel("Data Sampleo.xlsx", sheet_name = 'customer_db')
    sampled_parkings = parkings.loc[np.random.choice(parkings.index, parking_size, replace=False),cols].copy().reset_index(drop = True)
    sampled_customers = customers.loc[np.random.choice(customers.index, sample_size, replace=False),cols].copy().reset_index(drop = True)


    sample_coords = pd.concat([node_0,sampled_parkings, sampled_customers]).rename(columns={'node': 'NODES', 'longitude':'X',  'latitude':'Y'}).reset_index(drop = True)

    sample_parameters = pd.concat([sample_coords.NODES, 
                                   pd.Series(data = range(1,n_trucks+1), name='TRUCKS'), 
                                   pd.Series(data = range(1,n_drones+1), name='DRONES'),
                                   pd.Series(name=''),
                                   pd.DataFrame({"NDEPOTS":{"0":parking_size},"NCUSTOMERS":{"0":sample_size},"NTRUCKS":{"0":n_trucks},"NDRONES":{"0":n_drones},"ETR":{"0":410},"EDR":{"0":41},"QTR":{"0":750},"QDR":{"0":5},"MAXDDR":{"0":10},"WDR":{"0":14}}).reset_index(drop=True)
                                   ], axis=1)
    sample_demand = pd.DataFrame({
                                  "NODO": sample_coords.NODES,
                                  "DEMANDA" : [0 for x in range(0,parking_size+1)] + np.ceil(np.random.normal(avg_demand, std_demand, sample_size)).tolist()     
                                    })
    
    coordinates = sample_coords.set_index('NODES')[['Y', 'X']].to_numpy()
    nodes = sample_coords['NODES'].unique()
    euclidean_dm = pd.DataFrame(cdist(coordinates, coordinates, metric=haversine), index=sample_coords['NODES'], columns=sample_coords['NODES'])
    times_df = pd.read_excel('time.xlsx', header = 0, index_col=0)
    truck_times = times_df.loc[times_df.index.isin(sample_coords['NODES']), nodes]
    drone_times = euclidean_dm*1000/(8*60)  # x 1000m / 8 m/s * 60 s/min
    real_distances = pd.read_excel('distances.xlsx', header = 0, index_col=0)
    manhattan_df = real_distances.loc[real_distances.index.isin(sample_coords['NODES']), nodes]


    folder_path = f'instances/C{sample_size}P{parking_size}T{n_trucks}D{n_drones}'

    # Check if the folder exists, and if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with pd.ExcelWriter(f'instances/C{sample_size}P{parking_size}T{n_trucks}D{n_drones}/C{sample_size}P{parking_size}T{n_trucks}D{n_drones}.xlsx', engine='xlsxwriter') as writer:
        sample_parameters.to_excel(writer, sheet_name='PARAMETROS', index = False)
        sample_demand.to_excel(writer, sheet_name='DEMANDA', index = False)
        sample_coords.set_index('NODES').to_excel(writer, sheet_name='COORDS')
        manhattan_df.to_excel(writer, sheet_name='MANHATTAN')
        truck_times.to_excel(writer, sheet_name='TIEMPOS_CAM')
        drone_times.to_excel(writer, sheet_name='TIEMPOS_DRON')
        euclidean_dm.to_excel(writer, sheet_name='EUCLI')
            

    xlsx_files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]

    ws = gams.GamsWorkspace(working_directory=folder_path)
    main_xml = os.path.join(folder_path, f"NEOS INSTRUCTION {objective}.xml")  # Replace with the path to your xml file
    main_gms = os.path.join(folder_path, f"MODEL {objective}.gms")
    for file in xlsx_files:
        instance_name = file.split('.')[0]
        instance_gen = f"""sets
                            i nodos
                            k Camiones
                            l Drones
                            alias(i,j)
                            ;
                            parameters
                            NDepots Numero de parqueaderos
                            NCustomers Numero de clientes
                            NDrones Numero de Drones
                            NTrucks Nuemro de Vehiculos
                            ETR Emision de los camiones
                            EDR Emision de los drones
                            QTR Capacidad de los camiones
                            QDR Capacidad de los drones
                            MAXDDR Autonomia del dron en distancia
                            WDR Peso del dron
                            DISTmh(i,j) MaTDist Manhattan
                            DISTec(i,j) MaTDist Euclidiana
                            TIMET(i,j) MatTemp Camiones
                            TIMED(i,j) MatTemp Drones
                            DEM(i) Demanda del nodo i
                            $onecho > tasks.txt
                            dset=i rng=PARAMETROS!A2 rdim=1
                            dset=k rng=PARAMETROS!B2 rdim=1
                            dset=l rng=PARAMETROS!C2 rdim=1
                            par=NDepots rng=PARAMETROS!E2 rdim=0
                            par=NCustomers rng=PARAMETROS!F2 rdim=0
                            par=NTrucks rng=PARAMETROS!G2 rdim=0
                            par=NDrones rng=PARAMETROS!H2 rdim=0
                            par=ETR rng=PARAMETROS!I2 rdim=0
                            par=EDR rng=PARAMETROS!J2 rdim=0
                            par=QTR rng=PARAMETROS!K2 rdim=0
                            par=QDR rng=PARAMETROS!L2 rdim=0
                            par=MAXDDR rng=PARAMETROS!M2 rdim=0
                            par=WDR rng=PARAMETROS!N2 rdim=0
                            par=DISTmh rng=MANHATTAN!A1 rdim=1 cdim=1
                            par=DISTec rng=EUCLI!A1 rdim=1 cdim=1
                            par=TIMET rng=TIEMPOS_CAM!A1 rdim=1 cdim=1
                            par=TIMED rng=TIEMPOS_DRON!A1 rdim=1 cdim=1
                            par=DEM rng=DEMANDA!A2 rdim=1
                            $offecho
                            $CALL GDXXRW input={instance_name}.xlsx output=in.gdx trace=3 @tasks.txt"""
        job = ws.add_job_from_string(dedent(instance_gen))
        job.run(create_out_db=False)
        with open(main_gms, 'w') as file:
            file.write(dedent(model))
        with open(main_xml, 'w') as file:
            requests = f"""<document>
                                <category>milp</category>
                                <solver>CPLEX</solver>
                                <inputMethod>GAMS</inputMethod>
                            
                                <email>carlosrodsal@unisabana.edu.co</email>

                                <model><![CDATA[{dedent(model)}]]></model>

                                <gdx><base64>{base64.b64encode(open(os.path.join(folder_path, f"in.gdx"), 'rb').read()).decode()}</base64></gdx>

                                <restart></restart>

                                <wantgdx></wantgdx>

                                <wantlst></wantlst>

                                <wantlog><![CDATA[yes]]></wantlog>

                                <comments><![CDATA[]]></comments>

                            </document>"""
            file.write(dedent(requests))

    # Specify the list of allowed file extensions
    allowed_extensions = [".gdx", ".xlsx",".xml", ".gms"]  # Add your desired extensions

    # Iterate through the files in the directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the file has an extension
        if os.path.isfile(file_path):
            _, file_extension = os.path.splitext(filename)

            # Check if the file extension is not in the allowed list
            if file_extension not in allowed_extensions or filename.startswith(('225', '_g')):
                os.remove(file_path)  # Delete the file

def big_sample():
    customer_sample = (10,20,30)
    parking_sample = (5,10,15)
    trucks_sample = (5,)
    drones_sample = (15,)

    for ncustomers in customer_sample:
        for nparkings in parking_sample:
            for ntrucks in trucks_sample:
                for ndrones in drones_sample:
                    sample_generator(objective='z1',sample_size=ncustomers, parking_size=nparkings, n_trucks=ntrucks, n_drones=ndrones)

big_sample()