# -*- coding: utf-8 -*-
"""
Este programa encuentra aproximaciones a las raizes de función introducida por consola mediante el método de 
Newton-Raphson
Created on Mon Dec 28 12:14:27 2020
@author: Jorge San José Villar 
"""

import sympy as sym # Este es el módulo que se encarga de calcular las derivadas 
import numpy as np

def NewtonRaphson_2D(f, tolerancia, center_range = 0, start_seed = -2):# tolerancia = error_dado
    """
    Esta función aplica el método de N-R en dos dimensiones para multiples semillas (valores iniciales). Cuando el error entre 
    un valor calculado y el anterior es menor que uno especificado para el proceso y añade la raiz a una lista. Las semillas se
    generan con un np.arange que va desde un valor especificado a ese mismo valor en negativo. Además el centro de estas 
    semillas puede ser cambiado, esencialmente sumando un número al conjunto de números. 
    
    Input: 
        -f: Debe ser una string. Una función que va a ser casteada a una función de Sympy. Utiliza una única variable 'x' para diferenciar la 
        función respecto de ella y recuerda utilizar 'E' para el número e, expresar explicitamente que quieres 
        multiplicar con '*', utilizar sin(x) en vez de sen(x) y utilizar paréntesis si utilizas funciones como seno y coseno
        
        -tolerancia: Para mejores resultados debe ser 10^n, (escrito como '0.0...01' con n ceros). Determina la precisión
        de las raíces obtenidas parando el proceso iterativo cuando el error calculado es menor que esta. Además en función
        de ella se redondea un set con las raices obtenidas (Ej: error=0.0001 -> redondeo hasta la cuarta unidad) ya que
        no tiene sentido que las raices obtenidas tengan más decimales que la tolerancia dada.
        
        -center_range: Tiene como valor predetenminado cero (0). Se le suma a la array de semillas para mover todas ellas a
        la izquierda o derecha en la linea de números
        
        -start_seed: Tiene como valor predeterminado -2. Es donde empieza y termina el np.arange utilizado como valores
        iniciales para el método. 
    Output:
        -roots: Una lista con las raices encontradas sin redondear. Habrá tantas como semillas haya. Pueden existir duplicados
        en ella 
        -roots_set: Un set con las raices redondeadas en función de la tolerancia. Se eliminan los duplicasdos
    """
    """hay_error = True # Una flag
    while hay_error: # Bloque para errores
        try:
            symf = sym.sympify(f)
        except SyntaxError:
            print("Ha ocurrido un error de sintaxis. Recuerda: Utiliza una única variable 'x' para diferenciar la función respecto de ella y recuerda utilizar 'E' para el número e, expresar explicitamente que quieres multiplicar con '*', utilizar sin(x) en vez de sen(x) y utilizar paréntesis si utilizas funciones como seno y coseno")
        except TypeError:
            print("Se ha introducido algún símbolo incorrecto. Recuerda que debes usar: /, *, -, +, ^ (ó **) y . (en vez de ,)")
        except NameError:
            print("Se ha introducido en la función una variable no definida. recuerda utilizar 'x' como variable respecto de la cual se va a diferenciar")
        else: 
            hay_error = False"""
            
    try:
        symf = sym.sympify(f)
        symfdiff = sym.diff(symf)
    except SyntaxError:
        print("Ha ocurrido un error de sintaxis. Recuerda: Utiliza una única variable 'x' para diferenciar la función respecto de ella y recuerda utilizar 'E' para el número e, expresar explicitamente que quieres multiplicar con '*', utilizar sin(x) en vez de sen(x) y utilizar paréntesis si utilizas funciones como seno y coseno. Revisa la documentación de esta función para más información. Código de error: SyntaxError")
    except TypeError:
        print("Se ha introducido algún símbolo incorrecto. Recuerda que debes usar: /, *, -, +, ^ (ó **) y . (en vez de ,). Revisa la documentación de esta función para más información. Código de error: TypeError")
    except NameError:
        print("Se ha introducido en la función una variable no definida. Recuerda utilizar 'x' como variable respecto de la cual se va a diferenciar. Revisa la documentación de esta función para más información. Código de error: NameError")
    except ValueError:
        print("Se ha introducido en la función una variable no definida. Para el número e debes utilizar E. Recuerda utilizar 'x' como variable respecto de la cual se va a diferenciar. Revisa la documentación de esta función para más información. Código de error: Value error")
    else:
        seeds = np.arange(start_seed, -start_seed, 1.01) + center_range 

        error_calculado = 1.0
        roots = list() 
        for i in seeds: 
            x_i = i
            cont = 0
            while tolerancia <= error_calculado:
                x_siguiente = x_i - (symf.subs(sym.Symbol("x"), x_i))/(symfdiff.subs(sym.Symbol("x"), x_i))
                error_calculado = abs((x_siguiente - x_i))
                x_i = x_siguiente
                cont += 1
                if cont == 10000: # Failswich para que no se produzca un bucle infinito en caso de que no converga
                    break
            error_calculado = 1.0
            roots.append(x_siguiente)
        roots_copy = roots[:] # Debemos hacer una copia de la lista para poder devolver una lista y un set
        for i in range(len(roots)): # Esta parte se encaragará de hacer el redondeamiento
            roots_copy[i] = round(roots[i], int(np.log10(tolerancia**(-1))))
        return roots, set(roots_copy) 