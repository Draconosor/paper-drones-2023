import gams
import os
from time import sleep

gams_path = r"C:\GAMS\45"
folder_path = "instances"  # Replace with the path to your folder
xlsx_files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]

ws = gams.GamsWorkspace(system_directory=gams_path, working_directory='instances')

# Open the file in write mode ('w') to edit it
file_path = r"instances\Generador Instancias.gms"  # Replace with the path to your text file

for file in xlsx_files:
    instance_name = file.split('.')[0]
    with open(file_path, 'w') as file:
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
par=DEM rng=DEMANDA!A2 rdim=1
$offecho
$CALL GDXXRW input={instance_name}.xlsx output={instance_name}.gdx trace=3 @tasks.txt"""
        file.write(model)
    job = ws.add_job_from_file('Generador Instancias.gms')
    job.run()
# Specify the list of allowed file extensions
allowed_extensions = [".gms", ".gdx", ".lst", ".xlsx"]  # Add your desired extensions

# Iterate through the files in the directory
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Check if the file has an extension
    if os.path.isfile(file_path):
        _, file_extension = os.path.splitext(filename)

        # Check if the file extension is not in the allowed list
        if file_extension not in allowed_extensions:
            os.remove(file_path)  # Delete the file