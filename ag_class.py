import random
poblacion = [8, 4, 7, 1, 7, 6]
print("Poblacion inicial: ", poblacion)
padres = poblacion[:3]
print("Padres seleccionados: ", padres)

hijos = []
for i in range(0, len(padres) - 1):
    for j in range(i + 1, len(padres)):
        padre1 = padres[i]
        padre2 = padres[j]
        hijo1 = (padre1 + padre2)
        hijo2 = (padre1 * padre2)
        hijos.append(hijo1)
        hijos.append(hijo2)
print("Hijos generados: ", hijos)

mutado = random.randint(0, len(hijos) - 1)
hijos[mutado] = int(str(hijos[mutado])[::-1])
print("Hijos después de mutación: ", hijos)
individuos = hijos
maximo = max(individuos)
print("Individuo más grande: ", maximo)

while maximo <= 10000:
    padres = individuos[:3]
    print("Padres seleccionados: ", padres)

    hijos = []
    for i in range(0, len(padres) - 1):
        for j in range(i + 1, len(padres)):
            padre1 = padres[i]
            padre2 = padres[j]
            hijo1 = (padre1 + padre2)
            hijo2 = (padre1 * padre2)
            hijos.append(hijo1)
            hijos.append(hijo2)
    print("Hijos generados: ", hijos)

    mutado = random.randint(0, len(hijos) - 1)
    hijos[mutado] = int(str(hijos[mutado])[::-1])
    print("Hijos después de mutación: ", hijos)

    individuos = hijos
    maximo = max(individuos)
    print("Individuo más grande: ", maximo)
