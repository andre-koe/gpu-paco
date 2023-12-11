
Wir unterscheiden zwischen **Task Level Parallelism** und **Data Level Parallelism**
Besser bekannt unter den Namen **SIMD** und **MIMD**

Single Instruction Multiple Data (SIMD):
- "Same Task on different data"

Multiple Instruction Multiple Data (MIMD):
- "Multiple Tasks on same or different data"

### Unterschied zwischen GPUs und CPUs

| Eigenschaft | GPUs                               | CPUs                                |
|-------------|------------------------------------|-------------------------------------|
| Hauptfokus  | Grafikverarbeitung und parallele Verarbeitung großer Datenmengen | Vielseitige Verarbeitung und sequenzielle Aufgaben |
| Architektur | Tausende von kleineren, effizienten Rechenkernen für parallele Aufgaben | Weniger, aber leistungsfähigere Kerne für sequenzielle Aufgaben |
| Verwendung  | 3D-Rendering, KI und maschinelles Lernen, wissenschaftliche Simulationen | Allgemeine Zwecke, Betriebssysteme, Business-Anwendungen |
| Speicher    | Oft mit dediziertem, hochgeschwindigkeits Video-RAM | Nutzt meist System-RAM, seltener eigener Cache |
| Parallelität | Hoch, optimiert für parallele Datenverarbeitung | Begrenzt, fokussiert auf einzelne oder wenige Aufgaben gleichzeitig |
| Energieverbrauch | Oft höher, besonders bei intensiver Nutzung | Generell niedriger, optimiert für Energieeffizienz |
| Programmierung | Spezielle APIs und Sprachen wie CUDA, OpenCL | Standard-Programmiersprachen wie C, C++, Java |

#### Einschub: Heterogeneous Computing

Der Begriff "**Heterogeneous Computing**" beschreibt Systeme, welche mehr als einen Prozessortypen verwenden. Diese Systeme gründen ihre Effizienz hinsichtlich ihrer Ausführungsperformanz als auch ihres Energieverbrauchs auf den Einsatz von spezialisierten Prozessoren für spezialisierte Aufgaben.

#### Grundlegende Ausführungsschritte eines CUDA Programms:

- Initialisierung der Daten auf der CPU
- Transfer der Daten vom CPU Kontext auf den GPU Kontext
- Kernel Launch mit benötigter Grid/Block Größe
- Transfer der Daten zurück vom GPU Kontext in den CPU Kontext
- Freigabe des allokierten Speichers auf der CPU und GPU

#### Grundlegende Bestandteile eines CUDA Programms:

- Host Code (main Funktion) - läuft auf der CPU
- Device Code - läuft auf der GPU

Der Host - Code ist für das Aufsetzen der Ausführungskonfiguration (grid/block size, die Datenübertragung etc.) verantwortlich.

Der Kernel Code wird dann parallel in der spezifizierten Konfiguration ausgeführt. 

#### Modifiers im vor Funktionsnamen geben an, auf welcher Hardware der Code läuft:

Wir können die folgenden Modifier benutzen um dem Compiler zu signalisieren wo welche Code bestandteile laufen sollen:

- `__global__`
- `__kernel__`
- `__device__`

Beispiel für die Verwendung eines Modifiers:

`__global__ void hello_cuda(int x)`

Jeder Kernel sollte den return type `void` haben.
Wenn Werte aus dem Kernel zurückgegeben werden sollen, muss der entsprechende Speicherbereich mit spezifischen CUDA runtime calls transferiert werden.

#### Kernel Launch

Der Aufruf eines Kernels aus dem Host-Code wird auch als Kernel Launch bezeichnet
`hello_cuda << <1,1> >> ();`

#### Hello CUDA

Simples Hello World Programm mit CUDA

```c++
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

/**
 * Kernel Code mit __global__ Modifier (Ausführung auf der GPU)
 */
__global__ void hello_cuda() 
{
	printf("Hello CUDA!\n");
}

/**
 * Host Code (Ausführung auf der CPU)
 */

int main() 
{
	hello_cuda <<<1,1>>> (); // Kernel Launch, was ist '<< <1,1> >>' ?  
	return 0;
}
```

**Achtung**: Bei einem Kernel Launch handelt es sich um einen asynchronen Funktionsaufruf!

Falls wir auf die Beendigung des Kernel Launch warten wollen, müssen wir den folgenden CUDA Funktionsaufruf verwenden.

`cudaDeviceSynchronize()`

Damit wird sichergestellt, dass der Host Code auf die Beendigung aller vorangegangenen Kernel Launches wartet.  Es handelt sich von der Grundidee um einen Synchronisationsmechanismus vergleichbar mit `join` in **rust** im Kontext von Multithreading.

Unser Code sieht damit wie folgt aus:

```c++
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

/**
 * Kernel Code mit __global__ Modifier (Ausführung auf der GPU)
 */
__global__ void hello_cuda() 
{
	printf("Hello CUDA!\n");
}

/**
 * Host Code (Ausführung auf der CPU)
 */

int main() 
{
	hello_cuda <<<1,1>>> (); // Kernel Launch, was ist '<< <1,1> >>' ? 
	cudaDeviceSynchronize(); 
	return 0;
}
```

Am Ende der Programmausführung das Aufräumen nicht vergessen 

`cudaDeviceReset()` 

Von der Grundidee vergleichbar mit `free(*irgendwas)` in C und C++ aber umfassender, da der gesamte Gerätezustand der GPU auf den Zustand vor der Initialisierung der Laufzeitumgebung zurückgesetzt wird, nicht nur der Speicher.

Die Erhaltene Ausgabe unseres Programms ist:
`Hello CUDA`

#### Bedeutung der Kernel Launch Parameter `<<<1,1>>>`

Wir haben im Kernel Launch Aufruf des obigen Codes die Kernel Launch Parameter wie folgt angegeben `<<<1,1>>>`. Diese Parameter bestimmen wie der Kernel auf der GPU ausgeführt wird.

Es seien die Kernel Launch Parameter als `<<<a, b>>>` gegeben, dann steht

- `a` für die Anzahl an Thread-Blöcken steht auf welchen der Code ausgeführt werden soll. Jeder Block kann dabei unabhängig voneinander auf einem separaten SM ausgeführt werden.
- `b` für die Anzahl an Threads Threads pro Block welche den Code ausführen sollen. 

Diese Parameter definieren die Parallele Struktur auf der GPU. Insgesamt wird unser Programm demnach von `a * b` Threads gleichzeitig ausgeführt, die Gesamtanzahl aller Thread Launches wird **Grid** genannt.

#### Schlüsselkonzepte Grid & Block

`Kernel_name<<<number_of_blocks, number_of_threads_per_block>>>(args)`

Mit `number_of_blocks * number_of_threads_per_block => grid`

Das Konzept von Thread-Blöcken existiert nur für die GPU Programmierung und dient der Verwaltung des Workloads und der Vermeidung von Performanceeinbußen. Eine direkte Entsprechung gibt es im Bereich der CPU Multithread Programmierung nicht. 

Eine lose Analogie zum besseren Verständnis kann zu Kubernetes Clustern gezogen werden wobei `a` der Anzahl an Knoten und `b` der Anzahl an Pods in jedem Knoten entspricht.

#### dim3 Und multidimensionale Grids

In unseren vorherigen Beispielen haben wir die Anzahl der Blöcke (`number_of_blocks`) und die Anzahl der Threads pro Block (`number_of_threads_per_block`) als einfache skalare Werte behandelt. CUDA bietet jedoch auch die Möglichkeit, den speziell für mehrdimensionale Konfigurationen entwickelten Typ `dim3` zu verwenden.

Bei `dim3` handelt es sich um einen CUDA-spezifischen Datentyp, der verwendet wird, um 3-dimensionale Vektoren zu definieren. Eine `dim3`-Variable wird dabei wie folgt initialisiert
- `dim3 variable_name(X, Y, Z)` 
initialisiert, wobei `X`, `Y`, und `Z` die Dimensionen in den drei Achsen darstellen. Wenn einige dieser Werte nicht angegeben werden, werden sie standardmäßig auf 1 gesetzt. Zum Beispiel initialisiert `dim3 variable_name(X, Y)` eine `dim3`-Variable mit den Werten X, Y, und 1.

Die Verwendung von `dim3` ermöglicht eine deutlich feingranularere Konfiguration der Grid- und Block-Dimensionen, als es mit skalaren Werten möglich wäre. Dies ist besonders vorteilhaft für Algorithmen, die von Natur aus mehrdimensional sind, wie Bildverarbeitungs- oder Matrixoperationen.

![[Pasted image 20231206194450.png]]

Das obige Bild zeigt eine 1-Dimensionale Grid und Block Konfiguration. Im Code würde das wie folgt aussehen:

```c++
dim3 block(4,1,1); // dim3 block(4)
dim3 grid(8,1,1); // dim3 grid(8) 

hello_cuda<<<grid, block>>>();
```

In diesem Fall würden insgesamt `8 * 4 = 32` Threads gestartet werden. 

Normalerweise werden die Variablen in Abhängigkeit voneinander bestimmt:

```c++
int nx, ny;
nx = 16;
ny = 4;

dim3 block(8, 2);
dim3 grid(nx / block.x, ny / block.y)
```

Damit können wir die Anzahl an `blocks` dynamisch zur Laufzeit berechnen.

#### Limitierung der Block und Grid Größe

Die Größe eines Blocks ist beschränkt, und liegt bei 1024 x 1024 x 64 für die Dimensionen (X, Y, Z) wobei ein Thread Block so allokiert werden sollte, dass folgende Bedingung gilt:
$$ X\times Y\times Z \leq 1024$$

Genau wie die Block Size ist auch die Grid Size beschränkt, diese liegen bei 2^31 - 1, 65535, 65535 für die Dimensionen (X, Y, Z)

Zusammengenommen kommen wir auf ein theoretisches maximum der Anzahl an auszuführenden Threads von:$$1024 \times  2^{31} - 1 \times 65536 \times 65536 = 2.194.728.419.327$$ (für GPUs mit CUDA Compute Capability 3.0 und höher)

**Beispielprogramm**

```c++

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

__global__ void hello_cuda() {
	printf("Hello CUDA world\n");
}

int main() {

	dim3 grid(64, 16);
	dim3 block((int) pow(2, 31) - 1, 65535, 65535);
	hello_cuda <<<block, grid>>> ();

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
		return -1;
	}

	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}
```

Da CUDA von sich aus keine Fehlermeldungen auf der Konsole ausgibt müssen wir uns um die Fehlerbehandlung selbst kümmern.

![[Pasted image 20231206210820.png]]

Das Programm wird zumindest erfolgreich ausgeführt.

#### Thread Organisation (threadIdx) I/II

In CUDA wird jedem Thread innerhalb eines Blocks eine eindeutige Thread-ID (`threadIdx`) zugewiesen. Diese ID ist vom Typ `dim3`, was bedeutet, dass sie bis zu drei Dimensionen (x, y, z) hat. Die Werte von `threadIdx` hängen von der Position des konkreten Threads innerhalb seines Thread-Blocks ab:

- `threadIdx.x` gibt die Position des Threads innerhalb der X-Dimension des Blocks an.
- `threadIdx.y` für die Y-Dimension.
- `threadIdx.z` für die Z-Dimension.

**Wichtig ist, dass `threadIdx` nur innerhalb eines Thread-Blocks eindeutig ist!**

Durch die Verwendung von `threadIdx`, in Verbindung mit Block-Informationen wie `blockIdx` und `blockDim`, kann die globale Position eines Threads in einem gesamten Grid berechnet werden. Dies ist essenziell, um zu bestimmen, auf welchen Daten jeder Thread arbeitet.


**Beispielprogramm zur Ermittlung der ThreadIds**

```c++
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

__global__ void print_threadIdx() {
	printf("Thread: threadIdx.x : %d, threadIdx.y : %d, threadIdx.z : %d\n",
		threadIdx.x, threadIdx.y, threadIdx.z);
}

int main() {

	int nx, ny;
	nx = 16;
	ny = 16;

	dim3 block(8,8);
	dim3 grid(nx / block.x, ny / block.y);

	print_threadIdx <<<grid, block>>> ();

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
		return -1;
	}

	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}
```

Wir erhalten bspw. die folgende Ausgabe jeweils für einen der 4 ausgeführten Blöcke:

>Thread: threadIdx.x : 0, threadIdx.y : 0, threadIdx.z : 0
>Thread: threadIdx.x : 1, threadIdx.y : 0, threadIdx.z : 0
>Thread: threadIdx.x : 2, threadIdx.y : 0, threadIdx.z : 0
>Thread: threadIdx.x : 3, threadIdx.y : 0, threadIdx.z : 0
>Thread: threadIdx.x : 4, threadIdx.y : 0, threadIdx.z : 0
>Thread: threadIdx.x : 5, threadIdx.y : 0, threadIdx.z : 0
>Thread: threadIdx.x : 6, threadIdx.y : 0, threadIdx.z : 0
>Thread: threadIdx.x : 7, threadIdx.y : 0, threadIdx.z : 0


#### Thread Organisation (blockIdx) II/II

Genau wie die zuvor beschriebene `threadIdx` hat jeder Block in dem die Threads verwaltet werden auch eine `blockIdx`. 
Der Aufbau der ``blockIdx`` ist identisch wie zuvor. Allerdings ermöglicht es uns die `blockIdx` nun im Zusammenspiel mit der `threadIdx` und der Dimension des Blocks nun Threads innerhalb eines aus mehreren Blöcken bestehenden Grids eindeutig zu identifizieren.
Die Berechnung der 'globalen' `x` Position eines Threads sieht dann wie folgt aus:

```c++
// Kernel Code
...
int globalX = blockIdx.x * blockDim.x + threadIdx.x; 
...
```

#### `blockDim` & `gridDim`
`blockDim` & `gridDim` sind ebenfalls vom Typ `dim3` wobei

- `blockDim` die Anzahl an Threads in den jeweiligen Dimensionen eines Threadblocks spezifiziert.
- `gridDim` die Anzahl an Blöcken in den jeweiligen Dimensionen eines Grids spezifiziert.

Die Ausgabe aller relevanten Informationen erreichen wir mit folgendem Code:

```c++
// Aufruf des Kernels wie zuvor
__global__ void print_details() {
	printf("GridDim: gridDim.x %3d | gridDim.y %3d | gridDim.z %3d\n"
		"BlockDim: blockDim.x %3d | blockDim.y %3d | blockDim.z %3d\n"
		"BlockIdx: blockIdx.x %3d | blockIdx.y %3d | blockIdx.z %3d\n"
		"ThreadIdx: threadIdx.x %3d | threadIdx.y %3d | threadIdx.z %3d\n",
		gridDim.x, gridDim.y, gridDim.z,
		blockDim.x, blockDim.y, blockDim.z,
		blockIdx.x, blockIdx.y, blockIdx.z,
		threadIdx.x, threadIdx.y, threadIdx.z);
}
```

>GridDim: gridDim.x       2 | gridDim.y     2 | gridDim.z   1
>BlockDim: blockDim.x   8 | blockDim.y   8 | blockDim.z   1
>BlockIdx: blockIdx.x       0 | blockIdx.y     0 | blockIdx.z   0
>ThreadIdx: threadIdx.x   3 | threadIdx.y   7 | threadIdx.z   0

#### Beispiel: Array Indizierung mittels `threadIdx`

Der folgende Code zeigt exemplarisch wie man `threadIdx` nutzen kann um ein Array zu indizieren

```c++
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void unique_idx_calc_threadIdx(int * input) {
	int tid = threadIdx.x;
	printf("threadIdx: %d, value: %d \n", tid, input[tid]);
}


int main() {

	int array_size = 8;
	int array_byte_size = sizeof(int) * array_size;
	int h_data[] = { 23, 9, 4, 53, 65, 12, 1, 33 };

	for (int i = 0; i < array_size; i++) {
		printf("%d ", h_data[i]);
	}
	printf("\n\n");

	int* d_data;
	cudaMalloc((void**)&d_data, array_byte_size);
	cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

	dim3 block(8);
	dim3 grid(1);

	unique_idx_calc_threadIdx << <grid, block >> > (d_data);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
		return -1;
	}

	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}
```

(Einige der oben genutzten Funktionen für den Datentransfer zwischen `device` und `host` werden später genauer erläutert)

Die Ausgabe sieht dann wie folgt aus:
![[Pasted image 20231207170235.png]]

**Frage**: Dieser Ansatz ist nicht empfehlenswert, warum nicht?
Antwort: `threadIdx` sind nur innerhalb eines Blocks eindeutig

Wenn wir die folgenden Kernel Launch Parameter angeben:
`unique_idx_calc_threadIdx <<<2,4>>>(d_data)`

Erhalten wir folgende Ausgabe:
![[Pasted image 20231207174111.png]]

Daher: Von vornherein die globale `threadIdx` (`gid`) verwenden
> `gid = blockIdx.x * blockDim.x + threadIdx.x`

Wobei die Komponente `blockIdx.? * blockDim.?` den Offset in jedem Thread Block liefert

**Frage**: Wie verhält es sich mit zwei oder drei dimensionalen Grids?
Antwort: Hierfür müssen wir die Berechnung des Index Anpassen

> `gid = gridDim.x * blockDim.x * blockIdx.y + blockIdx.x * blockDim.x + threadIdx.x*`

Wobei es sich bei 
- `gridDim.x * blockDim.x * blockIdx.y` um den Zeilen Offset (Row offset)
- `blockIdx.x * blockDim.x` um den Block Offset
- `threadIdx.x` um die eindeutige Thread ID innerhalb eines Blocks
handelt

Code Beispiel für ein 2x2 Grid mit jeweils (2 Zeilen und 2 Spalten mit je einem Block mit jeweils 4 Threads)

```c++
__global__ void unique_idx_calc_threadIdx(int * input) {
	int tid = threadIdx.x;
	int block_offset = blockIdx.x * blockDim.x;
	int row_offset = gridDim.x * blockDim.x * blockIdx.y;
	int gid = row_offset + block_offset + tid;
	printf("threadIdx: %d, value: %d \n", gid, input[gid]);
}


int main() {

	int array_size = 16;
	int array_byte_size = sizeof(int) * array_size;
	int h_data[] = { 23, 9, 4, 53, 65, 12, 1, 33, 83, 46, -12, 99, 2, 19, 31, 102 };

	for (int i = 0; i < array_size; i++) {
		printf("%d ", h_data[i]);
	}
	printf("\n\n");

	int* d_data;
	cudaMalloc((void**)&d_data, array_byte_size);
	cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

	dim3 block(4);
	dim3 grid(2, 2);

	unique_idx_calc_threadIdx << <grid, block >> > (d_data);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
		return -1;
	}

	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}
```

Allgemein: Je komplexer das Grid, desto komplexer die Berechnung der `gid`

Die Art und Weise wie wir unsere `gid` berechnen bestimmt unser Memory Access Pattern und damit die Reihenfolge der Berechnung.

Letztes Beispiel Berechnung der `gid` für ein Grid 2 x 2 mit jeweils zwei 2 x 2 Blöcken

> `int tid = threadIdx.y * blockDim.x + threadIdx.x;`
> `int block_offset = blockDim.x * blockDim.y * blockIdx.x;`
> `int row_offset = gridDim.x * blockDim.x * blockDim.y * blockIdx.y;`
> `int gid = row_offset + block_offset + tid;`

#### Datentransfer zwischen Host und Device

![[Pasted image 20231207213030.png]]

Für den Transfer von Daten zwischen Host und Device verwenden wir den bereits gesehenen CUDA API Aufruf

`cudaMemCpy(*destination, *source, size in byte, direction)`

- Über die `direction` (`cudaMemCpyKind`) geben wir an von wo nach wo wir die Daten übertragen möchten
	- Device to Device - `cudamemcpydtod`
	- Device to Host - `cudamemcpydtoh`
	- Host to Device -  `cudamemcpyhtod`

Um Daten zu transferieren muss der dafür benötigte Speicherplatz wie aus der C/C++ Programmierung bekannt allokiert und freigegeben werden.
Die CUDA API bietet hierfür die folgenden an C/C++ orientierten Funktionen an, welche dies auf dem Device für uns übernehmen.

| C Funktion | CUDA Funktion | Beschreibung                                             |
|------------|---------------|----------------------------------------------------------|
| malloc     | cudaMalloc    | Reserviert Speicher auf dem Heap (Host vs. GPU)          |
| free       | cudaFree      | Gibt reservierten Speicher frei (Host vs. GPU)           |
| memcpy     | cudaMemcpy    | Kopiert Speicher zwischen Host und GPU oder innerhalb derselben |
| memset     | cudaMemset    | Setzt Speicher auf einen bestimmten Wert (Host vs. GPU)  |


Beispielcode: Kopieren und Ausgeben eines Arrays über das Device

```c++
__global__ void mem_transfer_test(int* input) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	printf("%d\n", input[gid]);
}


int main() {

	int array_size = 128;
	int array_byte_size = sizeof(int) * array_size;

	int* h_input;
	h_input = (int*)malloc(array_byte_size);

	time_t t;
	srand((unsigned)time(&t));

	for (int i = 0; i < array_size; i++) {
		h_input[i] = (int) (rand() & 0xff);
	}

	int* d_input;

	cudaMalloc((void**)&d_input, array_byte_size);
	cudaMemcpy(d_input, h_input, array_byte_size, cudaMemcpyHostToDevice);

	dim3 block(64);
	dim3 grid(2);

	mem_transfer_test << <grid, block >> > (d_input);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
		return -1;
	}

	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}
```

Einschub: Was machen wenn die Grid größe nicht zur Problemgröße passt?

Angenommen wir arbeiten mit einer festen Blocksize von 32 und unser Array hat 150 Elemente.
Wir initialisieren die Gridsize mit 5 denn, `5*32 > 150`
Aber wie verhindern wir das lesen von ungültigen Speicherbereichen durch Aufrufe der Art 
`input[149 + x] mit x > 0`
Einfach: Wir übergeben die array size als Aufrufparameter und führen eine Überprüfung durch


```c++
__global__ void mem_transfer_test(int* input, int array_size) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if (gid < array_size) {
		printf("%d\n", input[gid]);
	}
}


int main() {

	int array_size = 128;
	int array_byte_size = sizeof(int) * array_size;

	int* h_input;
	h_input = (int*)malloc(array_byte_size);

	time_t t;
	srand((unsigned)time(&t));

	for (int i = 0; i < array_size; i++) {
		h_input[i] = (int) (rand() & 0xff);
	}

	int* d_input;

	cudaMalloc((void**)&d_input, array_byte_size);
	cudaMemcpy(d_input, h_input, array_byte_size, cudaMemcpyHostToDevice);

	dim3 block(64);
	dim3 grid(2);

	mem_transfer_test << <grid, block >> > (d_input, array_byte_size);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
		return -1;
	}

	cudaDeviceSynchronize();

	cudaDeviceReset();
	return 0;
}
```


#### Beispiel Code: Summe dreier Arrays

```c++
__global__ void sum_arrays(int* arr1, int* arr2, int* arr3, int size, int* output) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	if (gid < size) {
		output[gid] = arr1[gid] + arr2[gid] + arr3[gid];
	}
}


int main() {

	int array_size = 10000;
	int array_byte_size = sizeof(int) * array_size;

	int *h_input_1, *h_input_2, *h_input_3, *gpu_res;

	// Allokieren von Speicherbereich auf dem host
	h_input_1 = (int*)malloc(array_byte_size);
	h_input_2 = (int*)malloc(array_byte_size);
	h_input_3 = (int*)malloc(array_byte_size);
	gpu_res = (int*)malloc(array_byte_size);

	time_t t;
	srand((unsigned)time(&t));

	for (int i = 0; i < array_size; i++) {
		h_input_1[i] = (int)(rand() & 0xff);
		h_input_2[i] = (int)(rand() & 0xff);
		h_input_3[i] = (int)(rand() & 0xff);
	}

	int* d_i_1, *d_i_2, *d_i_3, *d_res;

	// Allokieren von Speicherbereich auf dem device
	cudaMalloc((void**)&d_i_1, array_byte_size);
	cudaMalloc((void**)&d_i_2, array_byte_size);
	cudaMalloc((void**)&d_i_3, array_byte_size);
	cudaMalloc((void**)&d_res, array_byte_size);

	cudaMemcpy(d_i_1, h_input_1, array_byte_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_i_2, h_input_2, array_byte_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_i_3, h_input_3, array_byte_size, cudaMemcpyHostToDevice);

	dim3 block(128);
	dim3 grid((int)ceil(10000 / block.x) + 1);

	sum_arrays << <grid, block >> > (d_i_1, d_i_2, d_i_3, array_size, d_res);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
		return -1;
	}

	/*
	* Wichtig: Kernel Launch ist asynchron, wir müssen auf das Ergebnis warten
	* bevor wir die Daten zurück auf den host kopieren
	*/
	cudaDeviceSynchronize();

	cudaMemcpy(gpu_res, d_res, array_byte_size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < array_size; i++) {
		printf("%d\n", gpu_res[i]);
	}

	// Allokierten Speicherbereich freigeben, sowohl auf dem device als auch dem host
	cudaFree(d_i_1);
	cudaFree(d_i_2);
	cudaFree(d_i_3);
	cudaFree(d_res);

	free(h_input_1);
	free(h_input_2);
	free(h_input_3);
	free(gpu_res);

	cudaDeviceReset();
	return 0;
}
```


#### Einschub: Fehlerbehandlung in CUDA

Grundsätzlich hat jeder CUDA Funktionsaufruf einen Rückgabewert vom Typ `cudaError_t`.
Dieser hat entweder den Wert `success` oder einen von vielen Möglichen Error Typen abhängig davon was genau Fehlgeschlagen ist. Die Error Msg. kann man sich mithilfe der Funktion
`cudaGetErrorString(error)` als String zurückgeben lassen.

Es ist empfohlene Praxis (und Industriestandard) eine Fehlerbehandlung nach jedem Funktionsaufruf durchzuführen (sollte aus C (Systemaufrufen) bekannt sein).

Um die Code Beispiele möglichst kurz zu halten wurde an vielen Stellen darauf verzichtet.


#### Überlegungen in Bezug auf GPU Programmierung

- Performance
- Stromverbrauch
- Platzverbrauch
- Kosten der 

---
#### Das CUDA Execution Model

Im Abschnitt über Threads haben wir festgehalten, dass sich die höchstmögliche theoretische Anzahl an parallel auszuführenden Threads allein aus der `blockDim` und der `gridDim` bestimmt.
$$no\_of_\_threads = 1024 * (2^{31}-1) * 65535 * 65535 = 2.305.843.008.139.952.128$$
Diese Zahl stellt jedoch eine theoretische Obergrenze dar, die weit über den praktischen Kapazitäten moderner GPU-Hardware liegt. Um ein realistischeres Bild der parallelen Verarbeitungskapazität einer GPU zu erhalten, müssen wir die Hardware-Beschränkungen betrachten:


```c++
void query_device() {
	int devNo = 0;     
	cudaDeviceProp iProp;     
	cudaGetDeviceProperties(&iProp, devNo);      
	
	printf("Device %d: %s\n", devNo, iProp.name);     
	printf("Anzahl der MP: %d\n", iProp.multiProcessorCount);     
	printf("Max Anzahl von Threads pro MP: %d\n", iProp.maxThreadsPerMultiProcessor);   
	printf("Warp-Größe: %d\n", iProp.warpSize);     
	printf("Warps pro MP: %d\n", iProp.maxThreadsPerMultiProcessor / iProp.warpSize); 
}
```


Beispielausgabe für eine NVIDIA GeForce RTX 3080 GPU:

```
Device 0: NVIDIA GeForce RTX 3080 
Anzahl der Multiprozessoren: 48 
Maximale Anzahl von Threads pro Multiprozessor: 1536 
Warp-Größe: 32 
Maximale Anzahl von Warps pro Multiprozessor: 48`
```

Daraus ergibt sich eine praktische Obergrenze für die Anzahl gleichzeitig ausführbarer Threads:

`48 Multiprozessoren * 1536 Threads pro Multiprozessor = 73.728`


#### Warps

**Warps und ihre Rolle in der Thread-Ausführung:**

Ein Warp in CUDA ist eine Gruppe von 32 konsekutiven Threads, die die grundlegende Einheit der Ausführung auf einem Streaming Multiprozessor (SM  bzw. CUDA Kern) darstellen. Bei der Ausführung eines CUDA-Kernels wird ein Threadblock in mehrere Warps aufgeteilt.
Diese Warps werden dann auf den Multiprozessoren der GPU ausgeführt. Die effektive Nutzung und Scheduling von Warps sind entscheidend für die Leistungsoptimierung in CUDA-Anwendungen.

Die tatsächliche Anzahl der gleichzeitig auf einer GPU ausführbaren Threads ist somit durch die Anzahl der Multiprozessoren und die Anzahl der pro Multiprozessor ausführbaren Warps bzw. Threads begrenzt.

Alle Threads innerhalb eines Warps werden nach dem **SIMT** Prinzip ausgeführt.

> `SIMT ist ein Spezialfall von SIMD und steht für Single Instruction Multiple Threads` 

Jeder Thread eines Warps hat eine eindeutige ID, die Warp ID ist nur innerhalb eines Blocks eindeutig

![[Pasted image 20231208220325.png]]

Auf dem obigen Bild sehen wir die Aufteilung zweier Threadblöcke mit je 40 Threads,

jeder dieser Blöcke wird in Warps zu je 32 Threads aufgeteilt, es ist an dieser Stelle anzumerken, das jeweils ein Threadblock auf einem einzigen Streaming Multiprozessor ausgeführt wird. Daher können auch die Warps keine Threads aus mehreren Blöcken umfassen. Das führt bei einer suboptimalen Aufteilung dazu, das wir mitunter nicht ausgelastete Warps haben, und damit Threads "verschwenden".

Selbst wenn nur ein Thread aktiv ist, wird dieser innerhalb eines Warps ausgeführt, d.h. 1 aktiver, 31 inaktive Threads.

Wir können daraus schließen, das die optimale Blocksize ei vielfaches von `32` sein muss.

**Warp Analyse**

Um die oben beschriebenen Aspekte zu verdeutlichen schauen wir uns folgenden Code an:

```c++
__global__ void print_details_of_warps() {
    int gid = blockIdx.y * gridDim.x * blockDim.x 
		    + blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = threadIdx.x / 32;
    int gbid = blockIdx.y * gridDim.x + blockIdx.x;

    printf("tid: %d, bid.x: %d, bid.y: %d, gid: %d, warp_id: %d, gbid: %d\n", threadIdx.x, blockIdx.x, blockIdx.y, gid, warp_id, gbid);
}

int main()
{
    dim3 grid, block;
    block = dim3(42);
    grid = dim3(2, 2);

    print_details_of_warps <<<grid, block>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
    
}
```

Der Code startet einen Kernel in einem 2 x 2 Grid mit Blöcken von jeweils 42 Threads, welche ihre Block- Warp- und Grid-Id ausgeben.

Auszug aus der Ausgabe:

```
tid: 32, bid.x: 1, bid.y: 0, gid: 74, warp_id: 1, gbid: 1
tid: 33, bid.x: 1, bid.y: 0, gid: 75, warp_id: 1, gbid: 1
tid: 34, bid.x: 1, bid.y: 0, gid: 76, warp_id: 1, gbid: 1
tid: 35, bid.x: 1, bid.y: 0, gid: 77, warp_id: 1, gbid: 1
tid: 36, bid.x: 1, bid.y: 0, gid: 78, warp_id: 1, gbid: 1
tid: 37, bid.x: 1, bid.y: 0, gid: 79, warp_id: 1, gbid: 1
tid: 38, bid.x: 1, bid.y: 0, gid: 80, warp_id: 1, gbid: 1
tid: 39, bid.x: 1, bid.y: 0, gid: 81, warp_id: 1, gbid: 1
tid: 40, bid.x: 1, bid.y: 0, gid: 82, warp_id: 1, gbid: 1
tid: 41, bid.x: 1, bid.y: 0, gid: 83, warp_id: 1, gbid: 1
tid: 32, bid.x: 0, bid.y: 0, gid: 32, warp_id: 1, gbid: 0
tid: 33, bid.x: 0, bid.y: 0, gid: 33, warp_id: 1, gbid: 0
tid: 34, bid.x: 0, bid.y: 0, gid: 34, warp_id: 1, gbid: 0
tid: 35, bid.x: 0, bid.y: 0, gid: 35, warp_id: 1, gbid: 0
tid: 36, bid.x: 0, bid.y: 0, gid: 36, warp_id: 1, gbid: 0
tid: 37, bid.x: 0, bid.y: 0, gid: 37, warp_id: 1, gbid: 0
tid: 38, bid.x: 0, bid.y: 0, gid: 38, warp_id: 1, gbid: 0
tid: 39, bid.x: 0, bid.y: 0, gid: 39, warp_id: 1, gbid: 0
tid: 40, bid.x: 0, bid.y: 0, gid: 40, warp_id: 1, gbid: 0
tid: 41, bid.x: 0, bid.y: 0, gid: 41, warp_id: 1, gbid: 0
tid: 0, bid.x: 0, bid.y: 0, gid: 0, warp_id: 0, gbid: 0
tid: 1, bid.x: 0, bid.y: 0, gid: 1, warp_id: 0, gbid: 0
tid: 2, bid.x: 0, bid.y: 0, gid: 2, warp_id: 0, gbid: 0
tid: 3, bid.x: 0, bid.y: 0, gid: 3, warp_id: 0, gbid: 0
tid: 4, bid.x: 0, bid.y: 0, gid: 4, warp_id: 0, gbid: 0
tid: 5, bid.x: 0, bid.y: 0, gid: 5, warp_id: 0, gbid: 0
tid: 6, bid.x: 0, bid.y: 0, gid: 6, warp_id: 0, gbid: 0
tid: 7, bid.x: 0, bid.y: 0, gid: 7, warp_id: 0, gbid: 0
tid: 8, bid.x: 0, bid.y: 0, gid: 8, warp_id: 0, gbid: 0
tid: 9, bid.x: 0, bid.y: 0, gid: 9, warp_id: 0, gbid: 0
tid: 10, bid.x: 0, bid.y: 0, gid: 10, warp_id: 0, gbid: 0
tid: 11, bid.x: 0, bid.y: 0, gid: 11, warp_id: 0, gbid: 0
tid: 12, bid.x: 0, bid.y: 0, gid: 12, warp_id: 0, gbid: 0
tid: 13, bid.x: 0, bid.y: 0, gid: 13, warp_id: 0, gbid: 0
tid: 14, bid.x: 0, bid.y: 0, gid: 14, warp_id: 0, gbid: 0
tid: 15, bid.x: 0, bid.y: 0, gid: 15, warp_id: 0, gbid: 0
tid: 16, bid.x: 0, bid.y: 0, gid: 16, warp_id: 0, gbid: 0
tid: 17, bid.x: 0, bid.y: 0, gid: 17, warp_id: 0, gbid: 0
tid: 18, bid.x: 0, bid.y: 0, gid: 18, warp_id: 0, gbid: 0
tid: 19, bid.x: 0, bid.y: 0, gid: 19, warp_id: 0, gbid: 0
tid: 20, bid.x: 0, bid.y: 0, gid: 20, warp_id: 0, gbid: 0
tid: 21, bid.x: 0, bid.y: 0, gid: 21, warp_id: 0, gbid: 0
tid: 22, bid.x: 0, bid.y: 0, gid: 22, warp_id: 0, gbid: 0
tid: 23, bid.x: 0, bid.y: 0, gid: 23, warp_id: 0, gbid: 0
tid: 24, bid.x: 0, bid.y: 0, gid: 24, warp_id: 0, gbid: 0
tid: 25, bid.x: 0, bid.y: 0, gid: 25, warp_id: 0, gbid: 0
tid: 26, bid.x: 0, bid.y: 0, gid: 26, warp_id: 0, gbid: 0
tid: 27, bid.x: 0, bid.y: 0, gid: 27, warp_id: 0, gbid: 0
tid: 28, bid.x: 0, bid.y: 0, gid: 28, warp_id: 0, gbid: 0
tid: 29, bid.x: 0, bid.y: 0, gid: 29, warp_id: 0, gbid: 0
tid: 30, bid.x: 0, bid.y: 0, gid: 30, warp_id: 0, gbid: 0
tid: 31, bid.x: 0, bid.y: 0, gid: 31, warp_id: 0, gbid: 0
```

Man sieht die Aufteilung des Blocks mit ID (0,0) in die Warps 1 und 2 und betrachte die Länge der jeweiligen Ausgaben.


#### Warp Divergence

Wenn Threads des selben Warps unterschiedliche Instruktionen ausführen, sprechen wir von **Warp Divergence**.

Beispielcode der Warp Divergence verursacht:

```c++
__global__ void this_causes_warp_divergence() {
	int tid = threadIdx.x;
	if (tid % 2 == 0) {
		// do something
	} else {
		// do something else
	}
}
```

50% der Threads des ausführenden Warps führen die **If**-Bedingung aus, die anderen 50% die **else** Bedingung.

![[Pasted image 20231209131042.png]]

Die divergierenden Pfade werden seriell ausgeführt, darunter leidet die Ausführungseffizienz.

**Wichtig:** Die Einführung von Control-Flow-Statements in unserem Kernel Code führt nicht automatisch zu **Warp Divergence**

Beispiel: 1D Threadblock mit 64 Threads

```c++
__global__ void no_warp_divergence() {
	int tid = blockIdx.x * threadIdx.x;

	if (tid / 32 < 1) {
		// all threads within a single warp will either end up
		// here ...
	} else {
		// or here 
	}
}
```

#### Berechnung der Branch Effizienz

**Branch Effizienz** gibt den Prozentsatz an divergierenden Branches in einem Kernel an.

$$\text{Branch Efficiency} = 100\% \times \frac{\#\text{Branches}- \#\text{Divergent\ Branches}}{\text{\#Branches}}$$

Für den Code in `no_warp_divergence` ergibt sich eine Branch Effizienz von 100%.

Für den Code in `this_causes_warp_divergence` hingegen eine Branch Effizienz von 50%.


#### Einschub: Profiling mit NVIDIA Nsight Compute

![[Pasted image 20231209153843.png]]

##### Resource Partioning & Latency Hiding

In CUDA werden die Ressourcen eines Streaming Multiprozessors (SM) auf die aktiven Warps aufgeteilt. Der Ausführungskontext jedes Warps umfasst:

- **Program Counter:** Jeder Warp hat seinen eigenen Program Counter, der den Fortschritt des Warps durch den Code verfolgt.
- **Register:** Jeder Thread innerhalb eines Warps hat Zugriff auf einen Satz von Registern. Die Anzahl der verfügbaren Register kann die Anzahl der gleichzeitig auf einem SM aktiven Warps beeinflussen.
- **Shared Memory:** Warps innerhalb desselben Blocks können auf gemeinsamen Speicher (Shared Memory) zugreifen, der für schnelle Datenübertragungen und Synchronisation innerhalb des Blocks genutzt wird.

Während der Lebensdauer eines Warps auf einem SM wird sein Ausführungskontext auf dem Chip gehalten.

Daher sind die Kosten von Kontextwechseln minimal

> Der Wechsel zwischen Warps auf einem **SM** ist sehr effizient, da kein umfangreicher Austausch von Registerwerten oder anderen Kontextinformationen zwischen Speicher und Prozessor erforderlich ist.

**CUDA** nutzt das schnelle Umschalten zwischen Warps, um Latenzzeiten zu überbrücken die durch Speicherzugriffe (global, cache oder shared memory) oder andere Verzögerungen enstehen. Während ein Warp wartet, kann der SM zu einem anderen Warp wechseln und weiterarbeiten.
Dies wird in **CUDA** als **Latency Hiding** bezeichnet und ist von Konzept und Zielen mit Scheduling Verfahren auf Betriebssystemebene vergleichbar.


Die Ressourcen (Speicher) - Intensität hat Einfluss auf die Anzahl an Warps und Threadblocks die simultan auf einem SM ausgeführt werden können.

#### Warp Scheduling

Der Warp Scheduler eines SM kategorisiert Warps wie folgt:

- Selected Warp (Warp in Ausführung)
- Stalled Warp (Nicht bereit zur Ausführung)
- Eligible Warp (Warp bereit zur Ausführung aber im Moment nicht ausgeführt)

Ein Warp wird unter den folgenden Bedingungen als 'Eligible' klassifiziert:

- Verfügbarkeit der Ressourcen wie Register oder Shared Memory
- Alle Argumente für den aktuellen Befehl dieses Warps müssen bereit sein.

Warp Scheduling, ähnlich wie das Scheduling auf einer CPU, dient dazu, die Latenzzeiten bei Speicherzugriffen zu minimieren und die Auslastung der GPU zu maximieren.

#### Latenzoptimierung in CUDA

In CUDA gibt es zwei Hauptarten von Latenzen, die optimiert werden können: Memory Latency und Arithmetic Latency.

- **Memory Latency:** Bezieht sich auf die Verzögerungen beim Zugriff auf den Speicher, insbesondere auf den globalen DRAM der GPU. Die Optimierung der Memory Latency kann durch Techniken wie das Zusammenfassen von Speicherzugriffen (Memory Coalescing), effiziente Nutzung des Shared Memory und Minimierung von Zugriffen auf den globalen Speicher erfolgen.
    
- **Arithmetic Latency:** Bezieht sich auf die Zeit, die für die Ausführung von Rechenoperationen benötigt wird. Diese Art von Latenz kann durch die effiziente Organisation von Berechnungen, die Vermeidung von Abhängigkeiten zwischen Operationen und die gleichmäßige Verteilung von Rechenlasten über die Threads hinweg optimiert werden.
    

Der Warp Scheduler wechselt zwischen verschiedenen Warps, um die Auslastung der SMs zu maximieren. Wenn ein Warp auf Speicherzugriffe oder das Abschließen von Rechenoperationen wartet, kann der Scheduler zu einem anderen Warp wechseln, der bereit zur Ausführung ist. Dieses "verstecken der Latenz" (Latency Hiding) ermöglicht es der GPU, die durch Wartezeiten entstehenden Leistungseinbußen zu minimieren und eine hohe Auslastung zu gewährleisten.

Aufgabe des Anwendungsprogrammierers ist es hierbei genügend Parallelität in Form von zu Verfügung stehenden Warps bereitzuhalten, um Latency Hiding durch den Scheduler zu ermöglichen.

Je nachdem welche Art der Latenz in einer Anwendung dominiert sprechen wir von:

- Memory Bound Application (Memory Latency)
- Computation Bound Application (Arithmetic Latency)

#### Occupancy (Schlüsselkonzept)

Occupancy ist das Verhältnis der aktiven Warps zur maximalen Anzahl an Warps.
Es handelt sich hierbei um einen wichtigen Leistungsindikator, der angibt wie vollständig ein SM mit Warps belegt ist. Dies ist insbesondere im Zusammenspiel mit Latency Hiding von Bedeutung und hilft bei der Maximierung der Hardware Nutzung.

Allerdings ist eine höhere Occupancy nicht immer gleichzusetzen mit einer höheren Performance.


$$\textrm{Occupancy} = \frac{\textrm{Active Warps}}{\textrm{max. Warps}}$$
**Beispielrechnung:**

Berechnung der maximalen Anzahl Warps

Kernel nutzt 48 Register pro Thread und 4096 Bytes Shared Memory pro Block.
Die Blocksize liegt bei 128 Threads.

>Zur Erinnerung: Die Maximale Anzahl an Warps ist Abhängig von dem Ressourcenverbrauch der Warps

$$\textrm{Register pro Warp} = 48 * 32 = 1536$$ 
Maximale Anzahl an Registern pro SM: 65536 (Hardware abhängig)
Liefert uns die maximale Anzahl an zulässigen Warps von:

$$\textrm{Allowed warps} = 65536 / 1536 = 42.67 $$
Die Warp Allocation Granularity liegt bei unserer Hardware bei 4

> Warp Allocation Granularity: Besagt das Warps immer in Gruppen von 4 Warps allokiert werden, selbst wenn tatsächlich nur ein Warp ausgeführt werden soll.

Daraus ergibt sich eine tatsächliche Anzahl an Allowed warps mit 40

Die Anzahl an Shared Memory pro SM liegen bei 102400 Bytes
Damit können wir die maximale Anzahl an aktiven Blocks bestimmen

$$\textrm{Active Blocks} = 102400 / 4096 = 25$$
Da unsere Blocks jeweils 128 Threads und damit 4 Warps beinhalten kommen wir auf 100 Warps pro SM, jedoch erlauben Hardware Einschränkungen maximal 48* Warps pro SM.
Dies bedeutet, unsere Anzahl an Allowed Warps wird nicht durch den Shared Memory Beeinflusst und liegt weiterhin bei 40.

Daraus ergibt sich nun die Occupancy als:

$$\textrm{Occupancy} = \frac{40}{48} \approx 0.83 = 83\% $$
Glücklicherweise bietet uns **NVIDIA Nsight Compute** einen interaktiven Occupancy Calculator
Der uns die Arbeit abnimmt.

![[Pasted image 20231209202536.png]]

#### Guidelines für Grid- und Blocksize:

- Blocksize sollte immer ein vielfaches der Warp Size sein d.h. `x * 32`
- Kleine Blocksizes (< 128) vermeiden, da zu kleine Blocksizes schnell zum hardwarebedingten Warp Limit für SM führen, ohne dass die vorhanden Ressourcen tatsächlich ausgenutzt werden.
- Die Anzahl an Blocks sollte deutlich über der Anzahl der vorhandenen SM's liegen um die Parallelität voll ausznutzen zu können.
- Experimentieren mit der Ausführungskonfiguration für konkrete Problemstellung, hierbei ist der bereits erwähnte **NVIDIA Nsight Compute** hilfreich

#### Parallele Reduktion als Synchronisierung

`cudaDeviceSynchronize()` ermöglicht eine globale Synchronisierung zwischen Host und Device (Blockiert die Ausführung auf dem Host bis alle Device Operationen abgeschlossen sind). Wird typischerweise im Host Code aufgerufen.

`syncthreads()` ermöglicht die Synchronisierung innerhalb eines Blocks, d.h. Warps warten auf die Beendigung anderer Warps innerhalb eines Blocks, bevor alle Warps den nachfolgenden Code Abschnitt betreten. Wird typischerweise im Device Code aufgerufen.

##### Parallele Reduktion

Das allgemeine Problem des Ausführens von kommutativen und assoziativen Operationen zwischen Vektoren wird **Reduktionsproblem** genannt

```c++
\\ Sequentielle Reduktion
int sum = 0;
for (int i = 0; i < size; i++) {
	sum += array[i];
}
```

**Schritte zur Parallelisierung der sequentiellen Reduktion**

- Aufteilung des Vektors in Chunks
- Jeder Chunk wird separat aufsummiert
- Die Teilergebnisse jedes Chunks werden aufsummiert

###### Neighbored pair approach

Wir werden die Summe eines jeden Blocks (Chunks) iterativ aufsummieren und dabei in jeder Iteration ausgewählte Elemente mit einem Nachbarn (mit bestimmtem Offset) aufsummieren.

Für die erste Iteration wird der Offset 1 gewählt, in jeder darauffolgenden wird er mit 2 multipliziert

Die Anzahl an threads die aktiv Arbeit verrichten wird dabei in jeder Iteration durch den Wert des Offets dividiert.


Beispielcode des oben Beschriebenen Ansatzes:

```c++
for (int offset = 1; offset < blockdim.x; offset *= 2) {
	if (tid % (2 * offset) == 0) {
		input[tid] += input[tid + offset];
	}
}
```
