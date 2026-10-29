
import time
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. FUNCION PRINCIPAL DE LA SIMULACION
# ============================================================

def simular(
    N=10_000,
    nu=0.01,
    pasos=1_000_000,
    semilla=42,
    numero_mediciones=30,
    max_reintentos=20
):
    """
    Simula un sistema de coalescencia y fragmentación.

    N:
        Fuerza total del sistema. Se conserva durante toda la simulación.

    nu:
        Probabilidad de que ocurra una fragmentación.

    pasos:
        Número de iteraciones de la simulación.

    numero_mediciones:
        Número aproximado de veces que se registra la distribución
        para estudiar cómo evoluciona el sistema.

    semilla:
        Permite repetir la simulación con los mismos resultados aleatorios.

    max_reintentos:
        Número máximo de intentos para seleccionar un grupo diferente
        antes de utilizar un método alternativo.

    Devuelve:
        tiempos: iteraciones en las que se tomaron mediciones.
        distribuciones: fuerzas y conteos de grupos en cada medición.
        grupos_finales: tamaños de los grupos al terminar.
        metricas: número de grupos y tiempo transcurrido en cada medición.
    """

    # Comprobamos que los parámetros tengan valores válidos.
    if N < 2:
        raise ValueError("N debe ser al menos 2.")
    if not 0 <= nu <= 1:
        raise ValueError("nu debe estar entre 0 y 1.")
    if pasos < 1:
        raise ValueError("pasos debe ser positivo.")
    if numero_mediciones < 1:
        raise ValueError("numero_mediciones debe ser positivo.")

    rng = np.random.default_rng(semilla)

    # --------------------------------------------------------
    # REPRESENTACIÓN DEL SISTEMA
    # --------------------------------------------------------
    # Cada grupo tiene un identificador y una lista de individuos.
    # Al principio existe un único grupo que contiene los N individuos.

    grupos = {0: list(range(N))}

    # owner[i] indica a qué grupo pertenece el individuo i.
    # Así podemos localizar rápidamente el grupo de un individuo.
    owner = np.zeros(N, dtype=np.int64)

    # Identificador que se asignará al siguiente grupo nuevo.
    siguiente_id = 1

    # --------------------------------------------------------
    # SELECCIONAR EL PRIMER GRUPO
    # --------------------------------------------------------
    def seleccionar_grupo():
        # Elegimos un individuo al azar entre los N individuos.
        # Por tanto, un grupo de tamaño s tiene probabilidad s/N
        # de ser seleccionado.
        individuo = rng.integers(N)
        return int(owner[individuo])

    # --------------------------------------------------------
    # SELECCIONAR UN SEGUNDO GRUPO DIFERENTE
    # --------------------------------------------------------
    def seleccionar_otro_grupo(gid):
        # Buscamos otro individuo al azar, pero rechazamos los
        # que pertenezcan al mismo grupo gid.
        #
        # Esto conserva la selección proporcional al tamaño,
        # condicionada a elegir un grupo diferente.

        for _ in range(max_reintentos):
            individuo = rng.integers(N)
            otro = int(owner[individuo])

            if otro != gid:
                return otro

        # Si muchos individuos pertenecen al mismo grupo,
        # los rechazos pueden ser frecuentes.
        # Como alternativa, elegimos entre los demás grupos
        # con probabilidades proporcionales a sus tamaños.

        ids = []
        pesos = []

        for otro, miembros in grupos.items():
            if otro != gid:
                ids.append(otro)
                pesos.append(len(miembros))

        if not ids:
            raise RuntimeError("No existe otro grupo seleccionable.")

        pesos = np.asarray(pesos, dtype=float)
        probabilidades = pesos / pesos.sum()

        return int(rng.choice(ids, p=probabilidades))

    # --------------------------------------------------------
    # REGISTRAR LA DISTRIBUCION DE TAMAÑOS
    # --------------------------------------------------------
    def registrar_distribucion():
        # Obtenemos el tamaño de cada grupo.
        tamanos = np.fromiter(
            (len(miembros) for miembros in grupos.values()),
            dtype=np.int64
        )

        # fuerzas: tamaños distintos que existen en el sistema.
        # conteos: número de grupos que tiene cada tamaño.
        #
        # Por ejemplo:
        # fuerzas = [1, 2, 4]
        # conteos = [8, 3, 1]
        #
        # Significa que hay 8 grupos de tamaño 1,
        # 3 grupos de tamaño 2 y 1 grupo de tamaño 4.

        fuerzas, conteos = np.unique(
            tamanos,
            return_counts=True
        )

        # Frecuencia relativa entre los grupos existentes.
        # P(s) = n_s / M, donde M es el número total de grupos.
        probabilidades = conteos / len(grupos)

        # Guardamos también los conteos enteros para evitar
        # reconstruirlos posteriormente a partir de decimales.
        return fuerzas, conteos, probabilidades

    # --------------------------------------------------------
    # DEFINIR CUANDO SE TOMAN LAS MEDICIONES
    # --------------------------------------------------------
    # Las mediciones se distribuyen aproximadamente de forma
    # logarítmica: hay más detalle al comienzo y también se
    # observan tiempos grandes de la simulación.

    checkpoints = np.unique(
        np.geomspace(
            1,
            pasos,
            num=min(numero_mediciones, pasos)
        ).astype(int)
    )

    tiempos = []
    distribuciones = []
    numero_grupos = []
    tiempo_transcurrido = []

    inicio = time.perf_counter()
    proxima_medicion = 0

    # ========================================================
    # BUCLE PRINCIPAL: UNA ITERACION POR PASO
    # ========================================================
    for paso in range(1, pasos + 1):

        # Seleccionamos el primer grupo y obtenemos su tamaño.
        gid = seleccionar_grupo()
        miembros = grupos[gid]
        s = len(miembros)

        # ----------------------------------------------------
        # FRAGMENTACION
        # ----------------------------------------------------
        if rng.random() < nu:

            # Un grupo de tamaño s se divide en s grupos
            # individuales de tamaño 1.
            #
            # Si s = 1, no cambia nada: el grupo ya es individual.

            if s > 1:
                grupos.pop(gid)

                for individuo in miembros:
                    grupos[siguiente_id] = [individuo]
                    owner[individuo] = siguiente_id
                    siguiente_id += 1

        # ----------------------------------------------------
        # COALESCENCIA
        # ----------------------------------------------------
        else:

            # Para fusionar dos grupos deben existir al menos dos.
            if len(grupos) > 1:

                # Seleccionamos un segundo grupo diferente,
                # también con probabilidad proporcional a su tamaño.
                otro_gid = seleccionar_otro_grupo(gid)

                miembros_1 = grupos[gid]
                miembros_2 = grupos[otro_gid]

                # Fusionamos la lista más pequeña en la más grande.
                # Esto reduce las actualizaciones necesarias en owner.
                if len(miembros_1) >= len(miembros_2):
                    grande_id = gid
                    pequeno_id = otro_gid
                else:
                    grande_id = otro_gid
                    pequeno_id = gid

                grandes = grupos[grande_id]
                pequenos = grupos.pop(pequeno_id)

                # Actualizamos la pertenencia de los individuos
                # del grupo pequeño y los añadimos al grupo grande.
                for individuo in pequenos:
                    owner[individuo] = grande_id

                grandes.extend(pequenos)

        # ----------------------------------------------------
        # REGISTRO DE LA EVOLUCION TEMPORAL
        # ----------------------------------------------------
        # No calculamos la distribución en cada iteración porque
        # eso aumentaría mucho el tiempo de ejecución.
        # Solo la calculamos en los pasos definidos en checkpoints.

        if (
            proxima_medicion < len(checkpoints)
            and paso == checkpoints[proxima_medicion]
        ):
            fuerzas, conteos, probabilidades = registrar_distribucion()

            tiempos.append(paso)

            distribuciones.append({
                "fuerzas": fuerzas,
                "conteos": conteos,
                "probabilidades": probabilidades
            })

            numero_grupos.append(len(grupos))

            tiempo_transcurrido.append(
                time.perf_counter() - inicio
            )

            proxima_medicion += 1

    # --------------------------------------------------------
    # COMPROBACIONES FINALES
    # --------------------------------------------------------
    # La suma de los tamaños debe seguir siendo N.
    assert sum(map(len, grupos.values())) == N

    # Verificamos que cada individuo esté registrado en el grupo
    # al que realmente pertenece.
    for gid, miembros in grupos.items():
        assert all(owner[i] == gid for i in miembros)

    grupos_finales = np.array(
        [len(miembros) for miembros in grupos.values()],
        dtype=np.int64
    )

    metricas = {
        "numero_grupos": np.array(numero_grupos),
        "tiempo_segundos": np.array(tiempo_transcurrido),
        "tiempo_total": time.perf_counter() - inicio
    }

    return tiempos, distribuciones, grupos_finales, metricas

#(grupos) guarda qué individuos pertenecen a cada grupo y (owner) permite identificar rápidamente el grupo de un individuo sin recorrer todos los grupos.

# ============================================================
# 2. DISTANCIA ENTRE DOS DISTRIBUCIONES
# ============================================================

def distancia_total_variacion(dist_a, dist_b):
    """
    Compara las frecuencias relativas P(s) de dos mediciones.

    D = (1/2) * suma_s |P_a(s) - P_b(s)|

    D = 0 significa que ambas distribuciones son iguales.
    Cuanto menor sea D, menor es el cambio entre mediciones.
    """

    # Extraemos los tamaños y las probabilidades de cada medición.
    a = dict(zip(
        dist_a["fuerzas"],
        dist_a["probabilidades"]
    ))

    b = dict(zip(
        dist_b["fuerzas"],
        dist_b["probabilidades"]
    ))

    # Unimos los tamaños que aparecen en cualquiera de las dos.
    soporte = set(a) | set(b)

    # Si un tamaño no aparece en una distribución,
    # su probabilidad para esa distribución es cero.
    return 0.5 * sum(
        abs(a.get(s, 0.0) - b.get(s, 0.0))
        for s in soporte
    )
#Esta función ayuda a evaluar si la distribución está cambiando con el tiempo. Si la distancia entre mediciones consecutivas permanece pequeña durante varias mediciones, hay evidencia de estabilización de \(P(s)\). Sin embargo, una distancia pequeña aislada no demuestra que el sistema haya alcanzado un estado estacionario.


# ============================================================
# 3. PARAMETROS Y EJECUCION
# ============================================================

N = 10_000
nu = 0.01
pasos = 1_000_000

# Ejecutamos la simulación.
# La semilla fija permite reproducir esta realización.
tiempos, distribuciones, grupos_finales, metricas = simular(
    N=N,
    nu=nu,
    pasos=pasos,
    semilla=42,
    numero_mediciones=30
)

# Mostramos un resumen de la simulación.
print("SIMULACION FINALIZADA")
print("---------------------")
print(f"N = {N}")
print(f"nu = {nu}")
print(f"Iteraciones = {pasos:,}")
print(f"Grupos finales = {len(grupos_finales):,}")
print(f"Fuerza total = {grupos_finales.sum():,}")
print(f"Tiempo total = {metricas['tiempo_total']:.2f} s")

#Muestra cuántos grupos quedan al final, comprueba que la fuerza total siga siendo \(N\) y registra el tiempo de ejecución. El número final de grupos depende de la evolución aleatoria del sistema; no debe fijarse de antemano.


# ============================================================
# 4. EVOLUCION TEMPORAL DE LA DISTRIBUCION
# ============================================================

plt.figure(figsize=(10, 7))

# Seleccionamos como máximo ocho mediciones para que la gráfica
# sea legible y las curvas no se amontonen demasiado.
indices = np.unique(
    np.linspace(
        0,
        len(tiempos) - 1,
        min(8, len(tiempos)),
        dtype=int
    )
)

for i in indices:

    # Recuperamos los tamaños de grupo y sus conteos exactos.
    s = distribuciones[i]["fuerzas"]
    ns = distribuciones[i]["conteos"]

    # Buscamos cuántos grupos de tamaño 1 existen.
    indice_unidades = np.where(s == 1)[0]

    # Si no hay grupos individuales, no se puede normalizar
    # esta medición respecto a n_1.
    if len(indice_unidades) == 0:
        continue

    n1 = ns[indice_unidades[0]]

    if n1 == 0:
        continue

    # La normalización n_s/n_1 permite comparar la forma
    # de las distribuciones en diferentes momentos.
    plt.loglog(
        s,
        ns / n1,
        marker="o",
        markersize=3,
        linewidth=1,
        label=f"{tiempos[i]:,} iteraciones"
    )

plt.xlabel("Fuerza del grupo s")
plt.ylabel(r"$n_s/n_1$")
plt.title(f"Evolucion temporal (N={N}, nu={nu})")
plt.grid(True, which="both", linestyle="--", alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("evolucion_distribucion.pdf", bbox_inches="tight")
plt.show()

#La gráfica muestra la evolución temporal de la distribución normalizada \(n_s/n_1\) para \(N=10\,000\) y \(\nu=0.01\). Se observa que, en las primeras 489 iteraciones, la distribución presenta una caída mucho más pronunciada, mientras que, a medida que aumenta el número de iteraciones, las curvas comienzan a aproximarse y a superponerse en buena parte del intervalo de fuerzas, lo que sugiere que la forma de la distribución tiende a estabilizarse. Para fuerzas pequeñas, la razón \(n_s/n_1\) disminuye al aumentar el tamaño del grupo; para fuerzas grandes, aparecen fluctuaciones y diferencias entre las curvas, especialmente porque existen pocos grupos de gran tamaño. Aunque las distribuciones de las últimas mediciones muestran similitudes, esta gráfica por sí sola no demuestra que se haya alcanzado un estado estacionario; es necesario contrastarla con la evolución de la distancia entre distribuciones y realizar simulaciones con distintas semillas para determinar si \(10^6\) iteraciones son suficientes.


# ============================================================
# 5. DISTANCIA ENTRE DISTRIBUCIONES CONSECUTIVAS
# ============================================================

# Comparamos P(s) de cada medición con la medición anterior.
distancias = np.array([
    distancia_total_variacion(
        distribuciones[i - 1],
        distribuciones[i]
    )
    for i in range(1, len(distribuciones))
])

plt.figure(figsize=(9, 5))

plt.plot(
    tiempos[1:],
    distancias,
    marker="o",
    markersize=4
)

plt.xscale("log")
plt.xlabel("Iteraciones")
plt.ylabel("Distancia de variacion total")
plt.title("Cambio entre mediciones consecutivas")
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("estacionariedad.pdf", bbox_inches="tight")
plt.show()

#El pico abrupto se debe, principalmente, a la condición inicial de la simulación y al mecanismo de fragmentación. En tu código, el sistema comienza con un único grupo que contiene los \(N=10\,000\) individuos. Mientras ocurre la coalescencia, no hay cambios porque no existe otro grupo con el cual fusionarse; además, si ocurre una fragmentación, esta divide el grupo seleccionado en \(10\,000\) grupos individuales de tamaño 1. Como la probabilidad de fragmentación es \(\nu=0.01\), tarde o temprano sucede este evento y la distribución cambia radicalmente: pasa de tener un grupo de tamaño \(10\,000\) a tener \(10\,000\) grupos de tamaño 1. Por eso, la distancia de variación total entre ambas distribuciones es exactamente 1, su valor máximo posible. El pico aparece alrededor de las primeras centenas de iteraciones porque las mediciones están distribuidas logarítmicamente y el evento ocurre aleatoriamente; no representa una inestabilidad numérica, sino una transición brusca propia de la inicialización del modelo.


# ============================================================
# 6. NUMERO DE GRUPOS Y TIEMPO TRANSCURRIDO
# ============================================================

fig, ax1 = plt.subplots(figsize=(9, 5))

# Eje izquierdo: número de grupos que existen en el sistema.
ax1.plot(
    tiempos,
    metricas["numero_grupos"],
    marker="o",
    markersize=3
)

ax1.set_xscale("log")
ax1.set_xlabel("Iteraciones")
ax1.set_ylabel("Numero de grupos")
ax1.grid(True, which="both", linestyle="--", alpha=0.3)

# Eje derecho: tiempo real transcurrido durante la simulación.
ax2 = ax1.twinx()

ax2.plot(
    tiempos,
    metricas["tiempo_segundos"],
    linestyle="--",
    marker="s",
    markersize=3
)

ax2.set_ylabel("Tiempo transcurrido (s)")

plt.title("Evolucion del numero de grupos y coste computacional")
fig.tight_layout()
plt.savefig("evolucion_grupos_tiempo.pdf", bbox_inches="tight")
plt.show()

#Permite observar cómo cambia el número de grupos y cuánto tiempo de cómputo se ha consumido en cada medición. El número de grupos puede aumentar por fragmentación y disminuir por coalescencia.


# ============================================================
# 7. RESUMEN NUMERICO DE LA EVOLUCION
# ============================================================

print("\nEVOLUCION TEMPORAL")
print("-" * 68)
print(
    f"{'Iteraciones':>15} "
    f"{'N. grupos':>15} "
    f"{'Tiempo (s)':>15} "
    f"{'Distancia TV':>15}"
)

for i, paso in enumerate(tiempos):

    # Para la primera medición no existe una medición anterior.
    if i == 0:
        distancia = np.nan
    else:
        distancia = distancias[i - 1]

    print(
        f"{paso:15,d} "
        f"{metricas['numero_grupos'][i]:15,d} "
        f"{metricas['tiempo_segundos'][i]:15.2f} "
        f"{distancia:15.5f}"
    )

#La tabla permite identificar en qué intervalo de iteraciones la distancia entre distribuciones empieza a ser pequeña y si se mantiene así. Es útil para comparar varias ejecuciones, pero no debes recomendar un número de iteraciones únicamente a partir de una fila de la tabla.


# ============================================================
# 8. DISTRIBUCION FINAL
# ============================================================

# Contamos cuántos grupos finales hay de cada tamaño.
valores_s, conteos = np.unique(
    grupos_finales,
    return_counts=True
)

plt.figure(figsize=(8, 6))

plt.loglog(
    valores_s,
    conteos,
    "o",
    markersize=4
)

plt.xlabel("Fuerza del grupo s")
plt.ylabel(r"Numero de grupos $n_s$")
plt.title(f"Distribucion final (N={N}, nu={nu})")
plt.grid(True, which="both", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("distribucion_final.pdf", bbox_inches="tight")
plt.show()

#Esta gráfica muestra el número de grupos \(n_s\) para cada tamaño \(s\) al final de la simulación. Si en una representación log-log aparece aproximadamente una región lineal, puede ser compatible con una ley de potencia en ese intervalo.


# ============================================================
# 9. COMPARAR SEMILLAS (PRUEBA ADICIONAL)
# ============================================================
# Esta prueba puede tardar bastante.
# Para empezar, usa menos iteraciones y después aumenta el valor.

semillas = [10, 20, 30]

resultados_semillas = []

for semilla in semillas:

    tiempos_prueba, dist_prueba, grupos_prueba, metricas_prueba = simular(
        N=10_000,
        nu=0.01,
        pasos=200_000,
        semilla=semilla,
        numero_mediciones=20
    )

    # Distancias entre mediciones consecutivas.
    d = [
        distancia_total_variacion(
            dist_prueba[i - 1],
            dist_prueba[i]
        )
        for i in range(1, len(dist_prueba))
    ]

    resultados_semillas.append({
        "semilla": semilla,
        "grupos_finales": len(grupos_prueba),
        "tiempo_total": metricas_prueba["tiempo_total"],
        "ultima_distancia": d[-1] if d else np.nan
    })

print("\nCOMPARACION ENTRE SEMILLAS")
print("-" * 65)

for r in resultados_semillas:
    print(
        f"Semilla: {r['semilla']:>3} | "
        f"Grupos finales: {r['grupos_finales']:>6,d} | "
        f"Tiempo: {r['tiempo_total']:>8.2f} s | "
        f"Ultima distancia TV: {r['ultima_distancia']:.5f}"
    )

#número de iteraciones recomendado para cada \(\nu\)
