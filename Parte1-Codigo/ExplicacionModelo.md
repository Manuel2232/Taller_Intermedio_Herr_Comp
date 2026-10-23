Arquitectura y Lógica Computacional de la Simulación de Guerra de Guerrillas

1. Introducción y Fundamentos del Modelo Estocástico

El objetivo de esta simulación computacional es reproducir la dinámica estocástica de formación, crecimiento y disolución de unidades de ataque en conflictos insurgentes y terrorismo global, tal como proponen Johnson et al. en los documentos de investigación de referencia. La premisa fundamental del modelo radica en que la guerra asimétrica no opera bajo estructuras militares jerárquicas rígidas, sino como un sistema complejo autoorganizado de células o grupos insurgentes que se unen (coalescencia) y se destruyen o disuelven (fragmentación) de forma continua.

A nivel macroscópico, este proceso continuo de fusión y ruptura alcanza un estado estacionario dinámico caracterizado por una distribución de tamaños de grupo que sigue una ley de potencias disrupción-frecuencia:

n_s \propto s^{-\alpha}

donde n_s representa el número de unidades de ataque de fuerza s, y el exponente universal se sitúa en \alpha \approx 2.5. Para simular numéricamente este fenómeno mediante un algoritmo de Monte Carlo de eventos discretos, es obligatorio respetar una restricción física inviolable: la fuerza total de la insurgencia N se mantiene estrictamente constante a lo largo de todo el tiempo de simulación.

\sum_{k=1}^{M} s_k = N

donde M es el número variable de grupos activos presentes en un instante dado.

2. Intuición Física y Demostración Probabilística

Para comprender la lógica que gobierna el código, resulta de gran utilidad recurrir al modelo mental de los N combatientes individuales organizados en una fila recta.

2.1 El Modelo de los N Unos y los Paréntesis

Imaginemos que la fuerza insurgente total N está representada por N puntos de fuerza o combatientes individuales (representados por el número 1). La estructura de grupos dentro de la insurgencia equivale simplemente a colocar paréntesis alrededor de estos combatientes:

* Estado inicial (t=0): Fragmentación máxima. Existen N grupos independientes de tamaño s=1: (1), (1), (1), (1), (1), \dots, (1)
* Estado intermedio (t > 0): Tras varias iteraciones de coalescencia, algunos grupos se unen formando unidades más grandes: (1, 1, 1, 1), (1, 1), (1), (1, 1, 1), (1), (1)

En este estado intermedio, el primer grupo tiene una fuerza s_1 = 4, el segundo s_2 = 2, el tercero s_3 = 1, el cuarto s_4 = 3, y así sucesivamente. La suma de todas las fuerzas contenidas en los paréntesis vuelve a ser exactamente N.

2.2 Selección Ponderada mediante "Tiro al Blanco" Uniforme

El algoritmo establece que la probabilidad de seleccionar una unidad de ataque i para iniciar una interacción debe ser estrictamente proporcional a su fuerza s_i:

P(\text{seleccionar unidad } i) = \frac{s_i}{N}

Físicamente, esto se justifica porque un grupo armado más grande realiza más operaciones, ocupa más territorio y genera más comunicaciones, lo que incrementa la probabilidad de que sea detectado o involucrado en una dinámica de fusión o ataque.

Desde la perspectiva computacional, este requisito de probabilidad ponderada se resuelve quitando visualmente los paréntesis y lanzando un dardo completamente al azar entre las N casillas de la fila. Dado que cada combatiente individual tiene una probabilidad uniforme de 1/N de ser alcanzado:

1. Un grupo grande de fuerza s_i = 400 ocupa 400 espacios en la fila de N casillas. La probabilidad acumulada de impactar cualquiera de sus casillas es 400/N.
2. Un grupo pequeño de fuerza s_j = 1 ocupa 1 solo espacio. Su probabilidad de impacto es de apenas 1/N.

Así, la selección proporcional a la fuerza se logra de forma directa eligiendo un número entero aleatorio uniforme entre 0 y N-1.

3. Diagnóstico de la Implementación Inicial y Errores de Simulación

En los primeros intentos de código se identificaron tres fallas conceptuales y de programación que impedían reproducir la física del problema:

3.1 Violación de la Conservación de Fuerza N

Al ejecutar la unión de dos grupos mediante concatenación de listas:

grupos[i] = list(np.concatenate((grupos[i], grupos[k])))


el grupo k se sumaba al grupo i, pero la posición original grupos[k] no se eliminaba de la población. Esto provocaba que la fuerza total de la insurgencia creciera de forma descontrolada (N \to \infty), destruyendo la condición de conservación.

3.2 Errores de Índice Dinámico (IndexError)

Conforme los grupos se unen, el número total de grupos independientes M disminuye. Generar un índice con i = np.random.randint(0, N) sobre la lista de grupos activos lanzaba inmediatamente un error de fuera de rango, pues los índices válidos de la lista ya no iban de 0 a N-1, sino de 0 a M-1.

3.3 Anidamiento Incorrecto en Fragmentación

Al fragmentar un grupo grande utilizando transformaciones de forma como reshape(-1, 1), en lugar de liberar los elementos como unidades independientes en la lista principal, se creaba una sublista anidada dentro del mismo índice i.

4. Estructura y Lógica del Código Optimizado

Para resolver estas limitaciones, existen dos estrategias de programación en Python que garantizan la conservación de N, la selección proporcional rigurosa y un rendimiento computacional óptimo.

Estrategia 1: Matriz de Pertenencia de Combatientes (dueno)

Esta estrategia es la traducción directa del modelo de los N combatientes alineados. Se define un arreglo unidimensional dueno de longitud fija N, donde la posición p \in [0, N-1] representa al combatiente p, y el valor dueno[p] almacena un identificador entero (ID) correspondiente al grupo al que pertenece.

import numpy as np

N = 10000        # Fuerza total de la insurgencia
nu = 0.01        # Probabilidad de fragmentación
pasos = 500000   # Iteraciones del algoritmo Monte Carlo

# Inicialmente, cada combatiente p es su propio grupo (ID = p)
dueno = np.arange(N, dtype=int)
siguiente_id_libre = N

for _ in range(pasos):
    # 1. Selección ponderada en tiempo O(1)
    p = np.random.randint(0, N)
    g_i = dueno[p]  # ID del grupo al que pertenece el combatiente p
    
    # 2. Transición de Estado: Fragmentación vs. Coalescencia
    if np.random.rand() < nu:
        # --- FRAGMENTACIÓN ---
        # El grupo g_i se destruye: sus miembros reciben IDs de grupo únicos e individuales
        miembros = np.where(dueno == g_i)[0]
        num_miembros = len(miembros)
        if num_miembros > 1:
            dueno[miembros] = np.arange(siguiente_id_libre, siguiente_id_libre + num_miembros)
            siguiente_id_libre += num_miembros
    else:
        # --- COALESCENCIA ---
        # Seleccionar un segundo combatiente q al azar
        q = np.random.randint(0, N)
        g_k = dueno[q]  # ID del grupo al que pertenece q
        
        if g_i != g_k:
            # Reasignación masiva: el grupo g_k es absorbido por g_i
            dueno[dueno == g_k] = g_i

# Extracción de la distribución final de tamaños s
_, tamanos_grupos = np.unique(dueno, return_counts=True)


Análisis de Operaciones Computacionales

Paso del Algoritmo	Código NumPy Correspondedor	Complejidad	Función Física
Selección i	p = np.random.randint(0, N); g_i = dueno[p]	\mathcal{O}(1)	Elige un combatiente al azar; su grupo g_i queda seleccionado con probabilidad s_i/N.
Decisión Evento	if np.random.rand() < nu:	\mathcal{O}(1)	Evalúa el parámetro de probabilidad de fragmentación \nu.
Fragmentación	dueno[miembros] = np.arange(...)	\mathcal{O}(s_i)	Rompe el grupo seleccionado; cada combatiente vuelve a ser una unidad individual.
Coalescencia	dueno[dueno == g_k] = g_i	\mathcal{O}(N) (vectorizado)	Todos los miembros del grupo g_k pasan a ser parte del grupo g_i.

Estrategia 2: Lista Dinámica de Tamaños de Grupo

En caso de trabajar con el tamaño explícito de los grupos s, la lista debe actualizarse eliminando y añadiendo elementos de forma estricta para no alterar la suma N:

import numpy as np

N = 10000
nu = 0.01
pasos = 500000

# Inicialmente N grupos de tamaño 1
grupos = [1] * N

for _ in range(pasos):
    # Calcular vector de probabilidades actualizadas s_k / N
    probabilidades = np.array(grupos) / N
    
    # Muestrear índice proporcional a la fuerza del grupo
    idx_i = np.random.choice(len(grupos), p=probabilidades)
    
    if np.random.rand() < nu:
        # --- FRAGMENTACIÓN ---
        tamano_i = grupos.pop(idx_i)          # Remover el grupo destruido
        grupos.extend([1] * tamano_i)         # Agregar tamano_i grupos de fuerza 1
    else:
        # --- COALESCENCIA ---
        idx_k = np.random.choice(len(grupos), p=probabilidades)
        if idx_i != idx_k:
            grupos[idx_i] += grupos[idx_k]     # Fusionar fuerza en el grupo i
            grupos.pop(idx_k)                 # Eliminar el grupo k absorbido

tamanos_grupos = np.array(grupos)


5. Análisis y Representación Gráfica de Resultados

Una vez completadas las iteraciones y alcanzado el estado estacionario dinámico, los datos resultantes se procesan para su análisis estadístico.

5.1 Histograma en Escala Lineal

Si se grafica la frecuencia de los tamaños de unidad s en ejes lineales (plt.hist o plt.bar), se observa una distribución extremadamente asimétrica:

1. Una barra masiva en s=1 y s=2 que concentra la abrumadora mayoría de las observaciones.
2. Una cola muy larga y plana que se extiende hacia valores altos de s (unidades grandes), pero con frecuencias visualmente indistinguibles de cero.

Esta representación lineal no permite apreciar la estructura matemática de los eventos de gran magnitud.

5.2 Gráfico Log-Log y Verificación de Ley de Potencias

Para evaluar la ley de potencias, se aplican logaritmos a ambos ejes (\log_{10}(n_s) vs. \log_{10}(s)). En este espacio transformado, la relación de potencia n_s = C \cdot s^{-\alpha} se convierte en una ecuación lineal:

\log_{10}(n_s) = \log_{10}(C) - \alpha \log_{10}(s)

donde la pendiente de la recta corresponde directamente al negativo del exponente universal -\alpha = -2.5. La coincidencia entre los puntos simulados por el método de Monte Carlo y la recta teórica de pendiente -2.5 confirma la validez de la implementación.
