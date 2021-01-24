# -*- coding: utf-8 -*-
"""
Autor: Jorge San José
"""
# importamos solo lo que necesitamos
from sympy import sympify, Matrix, transpose, hessian, det
from tkinter import Tk, LabelFrame, Label, Entry, Button, W
from numpy import log10

def NewtonRaphson_2D_Optimos(f, tolerancia=0.001, max_iter=10, vectores_iniciales=[(1, 1), (-1, -1)]):
    """
    Lorem ipsum
    
    Input: 
        -f1: Debe ser una string. Una función que va a ser casteada a una función de Sympy. Utiliza una única variable 'x' 
        e 'y' para diferenciar la función respecto de ella y recuerda utilizar 'E' para el número e, expresar 
        explicitamente que quieres multiplicar con '*', utilizar sin(x) en vez de sen(x) y utilizar paréntesis si 
        utilizas funciones como seno y coseno
        
        -tolerancia: Para mejores resultados debe ser 10^n, (escrito como '0.0...01' con n ceros). Determina la precisión
        de las raíces obtenidas parando el proceso iterativo cuando el error calculado es menor que esta. Además en función
        de ella se redondea un set con las raices obtenidas (Ej: error=0.0001 -> redondeo hasta la cuarta unidad) ya que
        no tiene sentido que las raices obtenidas tengan más decimales que la tolerancia dada.
        
        -max_iter: Un número entero que representa el número máximo de iteraciones que la función calculará. Si se supera
        la función para los cálculos y devuelve la aproximación que tenga en ese momento
        
        -vectores_iniciales: Una lista de pares de numeros que representan estimaciones iniciales del punto de corte 
    Output:
        - vectores_solucion: Una lista de vectores que se han encontrado
        
        - vectores_que_convergen: Una lista que forma una correspondencia de 1-1 con los vectores introducidos (vectores_iniciales).
        Se trata de booleanos que dicen si el vector correspondiente converge (true) o no (false)
        
        - tipos_de_extremos: una lista que contiene mediante palabras de que tipo de extremo se trata. Lo hace de una
        forma similar a vectores_que_convergen
    """
    # Preparatorio
    
    # Casteamos las funciones a expresiones simbólicas de sympy y la ponemos en una matriz
    symf = sympify(f)
    F_matriz = Matrix([symf]) # la matriz 1x1 de la función f
    
    # Creamos el jacobiano y el hessiano de la función anterior
    V = Matrix([sympify("x"), sympify("y")]) # Esta es la que contiene las variables en función de las 
    # cuales se toman las derivadas parciales, es necesáreo porque así está hecha la función .jacobian
    J = transpose(F_matriz.jacobian(V))
    H = hessian(F_matriz, (V))
    
    # Creamos la lista que contendrá los extremos encontrados, otra lista que tendrá si el vector convergen o no y otra
    # con qué tipo de extremo es
    vectores_solucion = list()
    vectores_que_convergen = list()
    tipos_de_extremos = list()
    
    # Iteramos sobre los vectores iniciales, aplicaremos el método a cada vector especificado para encontrar un mayor número
    # de extremos
    for vector_inicial in vectores_iniciales: # por cada uno de los vectores iniciales haremos el N-R
        # r0 es  el vector 'i' de los vectores iniciales y r_siguiente es por ahora nulo
        r0 = Matrix(vector_inicial) 
        r_siguiente = Matrix([0, 0])
        
        # Sustituimos r0 en la matriz f1_f2_matrix de las funciones y (r0) en el jacobiano
        J_en_r = J.subs([[sympify("x"), r0[0]], [sympify("y"), r0[1]]])
        H_en_r = H.subs([[sympify("x"), r0[0]], [sympify("y"), r0[1]]])
        
        # Calculamos la solución del sistema lineal con los coeficientes de la matriz del jacobiano en r0 y con los terminos 
        # indepencientes de la función en r0
        try:
            solucion = H_en_r.gauss_jordan_solve(J_en_r)[0] # la fución devuelve dos matrices la primera tiene las  
        # soluciones y la segunda tiene las incognitas en caso de que no pueda resolver el sistema completamente, como 
        # siempre va a poder resolver el sistema solo necesitamos la primera entera
        except ValueError:
            vectores_que_convergen.append(False)
            pass
        else:
            # Actualizamos el valor de r_siguiente
            r_siguiente = r0 - solucion

            # Declaramos 'error_calculado' que es la variable que guardará el error en cada iteración y 'iteración' que 
            # guardará porqué iteración vamos
            error_calculado = 1.0
            iteracion = 0

            # Bucle iterativo de newton raphson 
            while tolerancia <= error_calculado or iteracion <= max_iter: # si hace demasiadas iteraciones parará el ciclo
                J_en_r = J.subs([[sympify("x"), r_siguiente[0]], [sympify("y"), r_siguiente[1]]])  # jacobiano sustituido
                H_en_r = H.subs([[sympify("x"), r_siguiente[0]], [sympify("y"), r_siguiente[1]]]) # Hessiano sustituido
                solucion = H_en_r.gauss_jordan_solve(J_en_r)[0] # La matriz de soluciones

                # Actualizamos el valor de r_siguiente
                r_siguiente = r0 - solucion # Hacemos los dos a la vez

                # Actualizamos el error
                error_calculado = abs(max([(r_siguiente[0]-r0[0]), (r_siguiente[1]-r0[1])]))

                # Actualizamos r0
                r0 = r_siguiente

                # Sumamos uno a la iteración 
                iteracion += 1
                #print(f"Iteración: {iteracion}  Error: {error_calculado.evalf()}")
                if abs(max(r_siguiente)) > 100000000: # En caso de que explote hacia el infinito y no converja
                    vectores_que_convergen.append(False)
                    break
            # Hacemos el redondeo de r_siguente con otro vector
            x_siguiente = round(r_siguiente[0].evalf(), int(log10(tolerancia**(-1))))
            y_siguiente = round(r_siguiente[1].evalf(), int(log10(tolerancia**(-1))))

            F_matriz_en_r = F_matriz.subs([[sympify("x"), r_siguiente[0].evalf()], [sympify("y"), r_siguiente[1].evalf()]])
            z_resultante = round(F_matriz_en_r[0].evalf(), int(log10(tolerancia**(-1))))

            # Añadimos el vector encontrado a la lista vectores_solucion
            vectores_solucion.append([x_siguiente, y_siguiente, z_resultante]) 
            vectores_que_convergen.append(True)
            
            # Comprobamos el hessiano para saber que tipo de extremo es
            if det(H_en_r) > 0:
                if H_en_r[0] > 0:
                    tipos_de_extremos.append("mínimo")
                else:
                    tipos_de_extremos.append("máximo")
            elif det(H_en_r) < 0:
                tipos_de_extremos.append("punto de silla")
            else:
                tipos_de_extremos.append("No se ha podio determinar")
    return vectores_solucion, vectores_que_convergen, tipos_de_extremos

def Pasar_parametros():   
    try:
        tolerancia = float(entrada_tolerancia.get()) + 0.000001
    except ValueError: 
        etiqueta_extremos["text"] = "Tolerancia no valida. Debes utilizar punto en vez de coma. Utiliza un número como 0.001 o 0.0001"
    else:
        try: 
            max_iter = int(entrada_max_iter.get())
        except ValueError:
            etiqueta_extremos["text"] = "Iteración máxima no válida"
        else: 
            try:
                vector_inicial = [(float(entrada_vx.get()), float(entrada_vy.get()))]
            except ValueError: 
                etiqueta_extremos["text"] = "Punto inicial no válido"
            else:
                try:
                    funcion = entrada_funcion.get()
                    extremos_y_convergecia = NewtonRaphson_2D_Optimos(f=funcion, tolerancia=tolerancia, max_iter=max_iter, vectores_iniciales=vector_inicial)
                except TypeError:
                    etiqueta_extremos["text"] = "Función no valida. Utiliza una única variable 'x' e 'y' para diferenciar la función respecto de ella y recuerda utilizar 'E' para el número e, expresar explicitamente que quieres multiplicar con '*', utilizar sin(x) en vez de sen(x) y utilizar paréntesis si utilizas funciones como seno y coseno"
                else:
                    extremos = extremos_y_convergecia[0]
                    vector_converge = extremos_y_convergecia[1]
                    tipo_de_extremo = extremos_y_convergecia[2]
                    
                    if vector_converge:
                        etiqueta_extremos["text"] = f"Se ha encontrado el extremo: {extremos}"
                        etiquta_que_tipo_de_extremo["text"] = f"El extremo encontrado es un: {tipo_de_extremo}"
                    else:
                        etiqueta_extremos["text"] = "El punto introducido no ha producido un sistema resoluble. Por favor introduzca otro punto"
                    
ventana = Tk() 
ventana.title("Encontrador de extremos mediante Newton-Raphson")

# hacemos un marco con todas las imputs
marco_inputs = LabelFrame(ventana, text="Entrada", padx=10, pady=10) 

# Definimos todo
etiqueta_funcion = Label(marco_inputs, text="Introduce la función: ")
etiqueta_tolerancia = Label(marco_inputs, text="Introduce la tolerancia: ")
etiqueta_max_iter = Label(marco_inputs, text="Introduce el número máximo de iteraciones: ")
etiqueta_v_inicial = Label(marco_inputs, text="Introduce las componentes el punto inicial estimación: \n(componente 'x' arriba e componente 'y' abajo)")

entrada_funcion = Entry(marco_inputs, width=35, borderwidth=5)
entrada_tolerancia = Entry(marco_inputs, width=35, borderwidth=5)
entrada_max_iter = Entry(marco_inputs, width=35, borderwidth=5)
marco_vector_inicial = LabelFrame(marco_inputs)

entrada_vx = Entry(marco_vector_inicial)
entrada_vy = Entry(marco_vector_inicial)

entrada_vx.pack()
entrada_vy.pack()

# Ponemos todo con .gird. Las etiquetas estarán pegadas a la izq.
etiqueta_funcion.grid(row=0, column=0, sticky=W)
etiqueta_tolerancia.grid(row=1, column=0, sticky=W)
etiqueta_max_iter.grid(row=2, column=0, sticky=W)

entrada_funcion.grid(row=0, column=1)
entrada_tolerancia.grid(row=1, column=1)
entrada_max_iter.grid(row=2, column=1)

etiqueta_v_inicial.grid(row=3, column=0)
marco_vector_inicial.grid(row=3, column=1)

boton_calc_extremos = Button(marco_inputs, text="Calcular con parámetros actuales", command=Pasar_parametros)

boton_calc_extremos.grid(row=4, column=0, columnspan=1)

marco_inputs.pack()

#Hacemos un marco con todas las outputs
marco_outputs = LabelFrame(ventana, text="Salida", padx=10, pady=10) 

etiqueta_extremos = Label(marco_outputs, text="Esperando a una función")
etiquta_que_tipo_de_extremo = Label(marco_outputs, text="")

etiqueta_extremos.grid(row=0, column=0)
etiquta_que_tipo_de_extremo.grid(row=1, column=0)

marco_outputs.pack()

ventana.mainloop()