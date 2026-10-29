
import time
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. SIMULACION OPTIMIZADA
# ============================================================

def simular(
    N=10_000,
    nu=0.01,
    pasos=1_000_000,
    semilla=42,
    numero_mediciones=25,
    max_reintentos=20
):
    """
    Simula coalescencia y fragmentacion de grupos.

    N: fuerza total conservada.
    nu: probabilidad de fragmentacion.
    pasos: numero de iteraciones.
    numero_mediciones: cantidad aproximada de registros temporales.
    max_reintentos: intentos de muestreo antes del metodo alternativo.

    Devuelve:
        tiempos: iteraciones de cada medicion.
        distribuciones: frecuencias relativas P(s) en cada medicion.
        grupos_finales: numero de grupos de cada fuerza al final.
        metricas: informacion temporal de la simulacion.
    """

    if N < 2:
        raise ValueError("N debe ser al menos 2.")
    if not 0 <= nu <= 1:
        raise ValueError("nu debe estar entre 0 y 1.")
    if pasos < 1:
        raise ValueError("pasos debe ser positivo.")

    rng = np.random.default_rng(semilla)

    # Cada grupo tiene un identificador y una lista de integrantes.
    grupos = {0: list(range(N))}

    # owner[i] indica el identificador del grupo del individuo i.
    owner = np.zeros(N, dtype=np.int64)
    siguiente_id = 1

    def seleccionar_grupo():
        # Elegir un individuo uniformemente equivale a elegir
        # un grupo con probabilidad proporcional a su tamaño.
        individuo = rng.integers(N)
        return int(owner[individuo])

    def seleccionar_otro_grupo(gid):
        """
        Selecciona otro grupo con probabilidad proporcional
        a su tamaño, excluyendo el grupo gid.

        El rechazo de individuos es exacto. Si resulta ineficiente,
        se utiliza un muestreo ponderado sobre los grupos restantes.
        """
        for _ in range(max_reintentos):
            individuo = rng.integers(N)
            otro = int(owner[individuo])

            if otro != gid:
                return otro

        # Alternativa para evitar demasiados rechazos cuando
        # un grupo concentra casi toda la fuerza.
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

    def registrar_distribucion():
        # Histograma de grupos por fuerza.
        tamanos = np.fromiter(
            (len(miembros) for miembros in grupos.values()),
            dtype=np.int64
        )

        fuerzas, cantidades = np.unique(
            tamanos, return_counts=True
        )

        # P(s): frecuencia relativa entre los grupos existentes.
        probabilidades = cantidades / cantidades.sum()

        return fuerzas, probabilidades

    # Mediciones espaciadas aproximadamente de forma logaritmica.
    checkpoints = np.unique(
        np.geomspace(
            1, pasos, num=min(numero_mediciones, pasos)
        ).astype(int)
    )

    tiempos = []
    distribuciones = []
    numero_grupos = []
    tiempo_transcurrido = []

    inicio = time.perf_counter()
    proxima_medicion = 0

    for paso in range(1, pasos + 1):

        # Seleccion de la primera unidad.
        gid = seleccionar_grupo()
        miembros = grupos[gid]
        s = len(miembros)

        if rng.random() < nu:
            # ------------------------------------------------
            # FRAGMENTACION
            # ------------------------------------------------
            # Una unidad de fuerza s se convierte en s
            # unidades independientes de fuerza 1.
            if s > 1:
                grupos.pop(gid)

                for individuo in miembros:
                    grupos[siguiente_id] = [individuo]
                    owner[individuo] = siguiente_id
                    siguiente_id += 1

        else:
            # ------------------------------------------------
            # COALESCENCIA
            # ------------------------------------------------
            # El segundo grupo se elige ponderando su tamaño.
            if len(grupos) > 1:
                otro_gid = seleccionar_otro_grupo(gid)

                miembros_1 = grupos[gid]
                miembros_2 = grupos[otro_gid]

                # Fusionar siempre la lista menor en la mayor
                # reduce el coste de actualizar owner.
                if len(miembros_1) >= len(miembros_2):
                    grande_id = gid
                    pequeno_id = otro_gid
                else:
                    grande_id = otro_gid
                    pequeno_id = gid

                grandes = grupos[grande_id]
                pequenos = grupos.pop(pequeno_id)

                for individuo in pequenos:
                    owner[individuo] = grande_id

                grandes.extend(pequenos)

        # ----------------------------------------------------
        # REGISTRO TEMPORAL
        # ----------------------------------------------------
        if (
            proxima_medicion < len(checkpoints)
            and paso == checkpoints[proxima_medicion]
        ):
            fuerzas, probabilidades = registrar_distribucion()

            tiempos.append(paso)
            distribuciones.append((fuerzas, probabilidades))
            numero_grupos.append(len(grupos))
            tiempo_transcurrido.append(
                time.perf_counter() - inicio
            )

            proxima_medicion += 1

    # Comprobaciones finales.
    assert sum(map(len, grupos.values())) == N
    assert np.all(owner >= 0)

    # Comprobar que cada individuo pertenece al grupo correcto.
    for gid, miembros in grupos.items():
        assert all(owner[i] == gid for i in miembros)

    grupos_finales = np.array(
        [len(miembros) for miembros in grupos.values()],
        dtype=np.int64
    )

    metricas = {
        "numero_grupos": np.array(numero_grupos),
        "tiempo_segundos": np.array(tiempo_transcurrido),
        "tiempo_total": time.perf_counter() - inicio,
    }

    return tiempos, distribuciones, grupos_finales, metricas


# ============================================================
# 2. MEDIDA DE CAMBIO ENTRE DISTRIBUCIONES
# ============================================================

def distancia_total_variacion(dist_a, dist_b):
    """
    Distancia de variacion total entre dos distribuciones:
        D = (1/2) * sum_s |P_a(s) - P_b(s)|

    D = 0 significa distribuciones identicas.
    Cuanto menor sea D, menor es el cambio entre mediciones.
    """
    sa, pa = dist_a
    sb, pb = dist_b

    a = dict(zip(sa, pa))
    b = dict(zip(sb, pb))

    soporte = set(a) | set(b)

    return 0.5 * sum(
        abs(a.get(s, 0.0) - b.get(s, 0.0))
        for s in soporte
    )


# ============================================================
# 3. EJECUTAR LA SIMULACION
# ============================================================

N = 10_000
nu = 0.01
pasos = 1_000_000

tiempos, distribuciones, grupos_finales, metricas = simular(
    N=N,
    nu=nu,
    pasos=pasos,
    semilla=42,
    numero_mediciones=30
)

print("SIMULACION FINALIZADA")
print("---------------------")
print(f"N = {N}")
print(f"nu = {nu}")
print(f"Iteraciones = {pasos:,}")
print(f"Grupos finales = {len(grupos_finales):,}")
print(f"Fuerza total = {grupos_finales.sum():,}")
print(f"Tiempo total = {metricas['tiempo_total']:.2f} s")



# ============================================================
# 4. EVOLUCION TEMPORAL: DISTRIBUCION NORMALIZADA COMO FIGURA S1
# ============================================================

plt.figure(figsize=(10, 7))

indices = np.unique(
    np.linspace(
        0, len(tiempos) - 1,
        min(8, len(tiempos)),
        dtype=int
    )
)

for i in indices:
    s, p_s = distribuciones[i]

    # Recuperar los conteos relativos originales:
    # p_s = n_s / M  =>  n_s = p_s * M
    M = metricas["numero_grupos"][i]
    ns = p_s * M

    # Normalizacion respecto al numero de grupos de fuerza 1.
    indice_unidades = np.where(s == 1)[0]

    if len(indice_unidades) == 0:
        continue

    n1 = ns[indice_unidades[0]]

    if n1 == 0:
        continue

    plt.loglog(
        s,
        ns / n1,
        marker="o",
        markersize=3,
        linewidth=1,
        label=f"{tiempos[i]:,} iteraciones"
    )

plt.xlabel("Fuerza de ataque s")
plt.ylabel(r"$n_s/n_1$")
plt.title(f"Evolucion temporal (N={N}, nu={nu})")
plt.grid(True, which="both", linestyle="--", alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("evolucion_distribucion.pdf", bbox_inches="tight")
plt.show()


# ============================================================
# 5. GRAFICA DEL CAMBIO ENTRE MEDICIONES
# ============================================================

distancias = np.array([
    distancia_total_variacion(
        distribuciones[i - 1], distribuciones[i]
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
plt.title("Cambio de la distribucion entre mediciones consecutivas")
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("estacionariedad.pdf", bbox_inches="tight")
plt.show()


# ============================================================
# 6. EVOLUCION DEL NUMERO DE GRUPOS Y TIEMPO DE EJECUCION
# ============================================================

fig, ax1 = plt.subplots(figsize=(9, 5))

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


# ============================================================
# 7. TABLA RESUMEN DE LAS MEDICIONES
# ============================================================

print("\nEVOLUCION TEMPORAL")
print("-" * 65)
print(
    f"{'Iteraciones':>15} {'N. grupos':>15} "
    f"{'Tiempo (s)':>15} {'Distancia TV':>15}"
)

for i, paso in enumerate(tiempos):
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


# ============================================================
# 8. DISTRIBUCION FINAL
# ============================================================

valores_s, conteos = np.unique(
    grupos_finales, return_counts=True
)

plt.figure(figsize=(8, 6))
plt.loglog(valores_s, conteos, "o", markersize=4)
plt.xlabel("Fuerza del grupo s")
plt.ylabel(r"Numero de grupos $n_s$")
plt.title(f"Distribucion final (N={N}, nu={nu})")
plt.grid(True, which="both", linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("distribucion_final.pdf", bbox_inches="tight")
plt.show()
