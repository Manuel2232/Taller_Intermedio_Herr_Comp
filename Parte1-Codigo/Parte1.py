import numpy as np
import matplotlib.pyplot as plt

# Parámetros del modelo (Figura S1 del archivo 0605035.pdf)
N = 10000          # Fuerza total constante de la insurgencia
nu = 0.01          # Probabilidad de fragmentación (1 en 100)
pasos = 1000000     # Iteraciones para alcanzar el estado estacionario


dueno = np.arange(N, dtype=int)
siguiente_id_libre = N

for _ in range(pasos):

    p = np.random.randint(0, N)
    g_i = dueno[p]  

    if np.random.rand() < nu:
        # --- FRAGMENTACIÓN ---

        miembros = np.where(dueno == g_i)
        num_miembros = len(miembros[0])
        if num_miembros > 1:
            dueno[miembros] = np.arange(
                siguiente_id_libre, siguiente_id_libre + num_miembros
            )
            siguiente_id_libre += num_miembros
    else:
        
        # --- COALESCENCIA ---

        q = np.random.randint(0, N)
        g_k = dueno[q]  # ID del grupo al que pertenece q
        if g_i != g_k:
            # Todos los miembros del grupo g_k se unen al grupo g_i
            dueno[dueno == g_k] = g_i

# Al finalizar, calculamos el tamaño 's' de cada grupo activo
ids_grupos, tamanos_grupos = np.unique(dueno, return_counts=True)

# Lista de [ID del grupo, capacidad]
datos_grupos = [
    [int(g), int(s)]
    for g, s in zip(ids_grupos, tamanos_grupos)
]

# Lista solo con las capacidades
lista_capacidades = tamanos_grupos.tolist()

print("Estado final alcanzado.")
print(f"Suma total de fuerza comprobada: {np.sum(tamanos_grupos)} (Debe ser exactamente N={N})")
print(f"Número de grupos activos: {len(tamanos_grupos)}")

# Contar la frecuencia de cada tamaño de grupo s
valores_s, conteos = np.unique(tamanos_grupos, return_counts=True)

plt.figure(figsize=(8, 6))
plt.loglog(valores_s, conteos / len(tamanos_grupos), 'o', alpha=0.6,
           label='Simulación Monte Carlo')

# Recta teórica con pendiente -2.5, anclada al primer punto de los datos
s_teorico = np.linspace(valores_s.min(), valores_s.max(), 200)
y_teorico = (conteos[0] / len(tamanos_grupos)) * (s_teorico / valores_s[0]) ** (-2.5)
plt.loglog(s_teorico, y_teorico, 'k--', label=r'Teórico ($\alpha = 2.5$)')

plt.xlabel('Fuerza de ataque (s)', fontsize=12)
plt.ylabel(r'Frecuencia relativa $n_s / N$', fontsize=12)
plt.title(f'Distribución de Unidades de Ataque (N={N}, $\\nu={nu}$)', fontsize=14)
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=11)
plt.savefig("distribucion_logaritmica.pdf", bbox_inches="tight")
plt.show()


# Option 2: Con plt.bar para conteos discretos exactos
valores_s, conteos = np.unique(tamanos_grupos, return_counts=True)

plt.figure(figsize=(9, 5))
plt.bar(valores_s, conteos, color='royalblue', edgecolor='black', alpha=0.8)

# Opcional: limitar el eje X para ver con más detalle la zona inicial (p. ej. hasta s=30)
# plt.xlim(0, 30)

plt.xlabel(r'Fuerza de ataque ($s$)', fontsize=12)
plt.ylabel('Número de unidades de ese tamaño', fontsize=12)
plt.title(f'Distribución de Tamaño de Unidades (Escala Lineal, $N={N}$, $\\nu={nu}$)', fontsize=13)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig("distribucion_lineal.pdf", bbox_inches="tight")
plt.show()