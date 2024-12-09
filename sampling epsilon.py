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


instances = [i for i in os.listdir('instances') if i.startswith('C')]

def constrained_instance(step: float, og_obj: float, instance: str, reslim = 7200):

    folder_path = os.path.join("instances", instance)
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
                    z2(k) Individual Makespan per Truck
                    binary variable x, y, v, s
                    integer variable u
                    free variables z1,z2
                    ;
                    scalar
                    epsz2 /{1-step}/
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
                    maxepsi(k) Limite tiempo maximo por vehiculo
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
                    maxepsi(k).. z2(k) =L= {og_obj}*epsz2;

                    Model Modelo1 /all/;
                    option MIP=CPLEX
                    option optcr=0.00000000000001;
                    modelo1.Reslim = {reslim};
                    set workmem 128;
                    set mip strategy file 2;
                    set mip limits treememory 10000;
                    Solve Modelo1 using mip minimizing z1;
                    Display x.L, y.L, v.L, s.L, z1.L, z2.L;
                    """
    main_gms = os.path.join(folder_path, f"MODEL z2 {int(round(step*100, 0))}pct.gms")
    main_xml = os.path.join(folder_path, f"NEOS INSTRUCTION z2 {int(round(step*100, 0))}pct.xml") 
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

objs = pd.read_excel('results z2.xlsx', index_col=0)

steps = (0.1,
         0.2,
         0.3,
         0.4,
         0.5)


for i in instances:
    my_obj = objs.loc[i, 'z2']
    for step in steps:
        instance_string = f'{i} {int(step*100)}'
        instances = pd.read_clipboard()
        c_reslim = instances.loc[instances['Instance'] == instance_string, 'Seconds'].values[0]
        constrained_instance(step, my_obj,i, reslim = round(c_reslim+10))