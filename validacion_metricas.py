import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import scipy.signal as signal
import matplotlib.pyplot as plt 

def calcular_metricas(sig1, sig2):
    # 1. Normalización de amplitud
    s1 = sig1 / np.max(np.abs(sig1)) if np.max(np.abs(sig1)) != 0 else sig1
    s2 = sig2 / np.max(np.abs(sig2)) if np.max(np.abs(sig2)) != 0 else sig2
    
    # Asegurar misma longitud
    n = min(len(s1), len(s2))
    s1 = s1[:n]
    s2 = s2[:n]
    
    # 2. Correlación Cruzada Normalizada
    correlacion_cruzada = signal.correlate(s1, s2, mode='full')
    corr_max = np.max(correlacion_cruzada) / (np.linalg.norm(s1) * np.linalg.norm(s2))
    
    # 3. Alineación temporal
    retardo = np.argmax(correlacion_cruzada) - (n - 1)
    if retardo > 0:
        s1_al = s1[retardo:]
        s2_al = s2[:-retardo]
    elif retardo < 0:
        s1_al = s1[:retardo]
        s2_al = s2[-retardo:]
    else:
        s1_al = s1
        s2_al = s2

    # Cálculos de error
    mse = np.mean((s1_al - s2_al) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(s1_al - s2_al))
    
    # Relación Señal-Ruido (SNR en dB)
    potencia_senal = np.mean(s1_al ** 2)
    snr = 10 * np.log10(potencia_senal / mse) if mse != 0 else float('inf')
        
    return mse, rmse, mae, corr_max, snr, len(s1_al), s1_al, s2_al

def comparar_archivos():
    root = tk.Tk()
    root.withdraw()
    
    path1 = filedialog.askopenfilename(
        title="Selecciona el primer archivo de señal (.txt)",
        filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")]
    )
    if not path1:
        return
        
    path2 = filedialog.askopenfilename(
        title="Selecciona el segundo archivo de señal (.txt)",
        filetypes=[("Archivos de texto", "*.txt"), ("Todos los archivos", "*.*")]
    )
    if not path2:
        return
        
    try:
        data1 = np.loadtxt(path1, skiprows=1)
        data2 = np.loadtxt(path2, skiprows=1)
        
        sig1 = data1[:, 1] if data1.ndim > 1 and data1.shape[1] >= 2 else data1.flatten()
        sig2 = data2[:, 1] if data2.ndim > 1 and data2.shape[1] >= 2 else data2.flatten()
       
        mse, rmse, mae, corr, snr, muestras, s1_al, s2_al = calcular_metricas(sig1, sig2)
        
        # <-- BLOQUE DE LA GRÁFICA MODIFICADO (EFECTO LUPA) -->
        tiempo_muestras = np.arange(muestras)
        
        # Usamos subplots para tener más control sobre los ejes
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Gráfica principal
        ax.plot(tiempo_muestras, s1_al, label='Señal Teórica (Original)', color='blue', linewidth=1.5)
        ax.plot(tiempo_muestras, s2_al, label='Señal Reconstruida (Acústica)', color='red', linestyle='--', linewidth=1.5)
        
        ax.set_title('Comparativa Temporal: Señal Teórica vs. Reconstrucción Acústica')
        ax.set_xlabel('Muestras')
        ax.set_ylabel('Amplitud Normalizada')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.7)
        
        # -- CONFIGURACIÓN DEL ZOOM (LUPA) --
        # Creamos los ejes de la lupa. Los valores son: [x, y, ancho, alto] en fracciones (0 al 1) respecto a la ventana.
        # [0.4, 0.15, 0.3, 0.3] lo coloca en la parte central inferior donde no tapa los picos importantes.
        axins = ax.inset_axes([0.10, 0.55, 0.25, 0.35])
        
        # Dibujamos las mismas líneas dentro de la lupa
        axins.plot(tiempo_muestras, s1_al, color='blue', linewidth=1.5)
        axins.plot(tiempo_muestras, s2_al, color='red', linestyle='--', linewidth=1.5)
        
        # Definimos los límites del zoom para apuntar al primer gran valle (Muestras 5 a 25 aprox)
        x1, x2 = 5, 25 
        y1, y2 = -0.65, -0.40 # Ajusta estos valores según veas hasta dónde baja tu señal
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        
        axins.grid(True, linestyle=':', alpha=0.5)
        axins.tick_params(labelsize=8) # Números más pequeños en la lupa
        
        # Dibuja las líneas diagonales conectando la lupa con la zona original
        ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1.5)
        # -----------------------------------
        
        plt.tight_layout()
        plt.show() 

        # Formatear resultados
        resultado = (
            f"--- MÉTRICAS DE COMPARACIÓN DE SEÑALES ---\n\n"
            f"• Muestras evaluadas: {muestras}\n"
            f"• Error Cuadrático Medio (MSE): {mse:.6e}\n"
            f"• Raíz del Error Cuadrático (RMSE): {rmse:.6f}\n"
            f"• Error Absoluto Medio (MAE): {mae:.6f}\n"
            f"• Coeficiente de Correlación: {corr:.4f}\n"
            f"• Relación Señal-Ruido (SNR): {snr:.2f} dB"
        )
        
        messagebox.showinfo("Resultados de la Comparación", resultado)
        print(resultado)
        
    except Exception as e:
        messagebox.showerror("Error", f"No se pudieron procesar los archivos:\n{str(e)}")

if __name__ == "__main__":
    comparar_archivos()