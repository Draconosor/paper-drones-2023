sets
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
par=DEM rng=DEMANDAS!A2 rdim=1
$offecho
$CALL GDXXRW <_io.TextIOWrapper name='instances\\Generador Instancias.gms' mode='w' encoding='cp1252'> trace=3 @tasks.txt
$GDXIN C50P10T5D10.gdx