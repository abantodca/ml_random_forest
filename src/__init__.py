"""Training service - paquete raiz.

Side-effect controlado: forzamos el backend de matplotlib a 'Agg' (headless)
ANTES de que cualquier modulo del proyecto importe pyplot. Sin esto, si el
primer modulo en cargar pyplot lo hace bajo un backend interactivo (Qt5,
TkAgg) e.g. via dependencia transitiva, `matplotlib.use('Agg')` posterior
queda como no-op silencioso y cualquier `plt.show()` en remote crashea.

Este es el unico lugar correcto para setearlo: cualquier `import src.*`
ejecuta este __init__ primero, asi que nunca llegamos tarde.
"""
import matplotlib

matplotlib.use("Agg")
