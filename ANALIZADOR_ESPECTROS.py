import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import TextBox, Button
from scipy.signal import windows, find_peaks
import warnings
import tkinter as tk
from tkinter import ttk, messagebox

# Ignorar advertencias de interfaz gráfica menores
warnings.filterwarnings("ignore")

# --- 1. VENTANA DE SELECCIÓN DE MICRÓFONO (TKINTER) ---
selected_device_id = None

def iniciar_analizador():
    global selected_device_id
    selection = combo_devices.get()
    if not selection:
        messagebox.showerror("Error", "Por favor, selecciona un micrófono de la lista.")
        return
    # Extraemos el ID numérico del texto del desplegable
    selected_device_id = int(selection.split(" - ")[0])
    root.destroy()  

root = tk.Tk()
root.title("UPCT - Seleccionar Micrófono")
root.geometry("480x190")
root.configure(bg="#222222")

tk.Label(root, text="Selecciona el dispositivo de entrada de audio:", fg="white", bg="#222222", font=("Arial", 11, "bold")).pack(pady=15)

# Obtener lista de dispositivos
input_devices = []
devices = sd.query_devices()
for i, dev in enumerate(devices):
    if dev['max_input_channels'] > 0: 
        input_devices.append(f"{i} - {dev['name']}")

combo_devices = ttk.Combobox(root, values=input_devices, width=55, state="readonly")
if input_devices:
    combo_devices.current(0)  
combo_devices.pack(pady=5)

btn_start = tk.Button(root, text="Iniciar Analizador", command=iniciar_analizador, bg="#00ffcc", fg="black", font=("Arial", 10, "bold"), relief="flat", cursor="hand2")
btn_start.pack(pady=15)

root.mainloop()

if selected_device_id is None:
    print("No se seleccionó ningún dispositivo. Saliendo...")
    exit()

# --- 2. CONFIGURACIÓN AUTOMÁTICA ---
device_info = sd.query_devices(selected_device_id, 'input')
FS = int(device_info['default_samplerate'])

FFT_SIZE = 8192
BUFFER_SIZE = 2048
F_MIN = 20  
F_MAX = 20000  

PEAK_HEIGHT_THRESHOLD = 0.005 
PEAK_PROMINENCE_THRESHOLD = 0.002
MIN_DISTANCE_BETWEEN_PEAKS = 10

# Límites por defecto de las gráficas
max_time_x = BUFFER_SIZE / FS
max_time_y = 1.0
max_freq_y = 0.5   
max_mag_db = 0.0   
current_f_max = F_MAX # NUEVO: Variable para controlar el límite X del espectro

resp_frecuencia_on = False
scale_mode = 'lin'      
mag_scale_mode = 'lin'  
is_frozen = False   
trigger_on = False  

plt.style.use('dark_background')

# --- VARIABLES GLOBALES ---
audio_buffer = np.zeros(FFT_SIZE)
pointer = 0

window = windows.blackmanharris(BUFFER_SIZE)
freqs_global = np.fft.rfftfreq(FFT_SIZE, d=1/FS)

current_magnitude = np.zeros(len(freqs_global))
max_magnitudes_visuales = np.zeros(len(freqs_global))

num_bins = 500
log_freqs = np.logspace(np.log10(F_MIN), np.log10(F_MAX), num_bins)
lin_freqs = np.linspace(F_MIN, F_MAX, num_bins)
waterfall_history = 100 
waterfall_data = np.full((waterfall_history, num_bins), -120.0)

# --- AUDIO CALLBACK ---
def audio_callback(indata, frames, time, status):
    global audio_buffer, pointer, current_magnitude, max_magnitudes_visuales
    
    indata_flat = np.mean(indata, axis=1) 
    num_samples = len(indata_flat)
    
    start_pos = pointer % FFT_SIZE
    end_pos = (pointer + num_samples) % FFT_SIZE
    if end_pos > start_pos:
        audio_buffer[start_pos:end_pos] = indata_flat
    else:
        audio_buffer[start_pos:] = indata_flat[:FFT_SIZE - start_pos]
        audio_buffer[:end_pos] = indata_flat[FFT_SIZE - start_pos:]
    pointer += num_samples

    p = pointer % FFT_SIZE
    if p >= BUFFER_SIZE:
        data_window = audio_buffer[p - BUFFER_SIZE : p]
    else:
        data_window = np.concatenate((audio_buffer[-(BUFFER_SIZE - p):], audio_buffer[:p]))
        
    windowed_data = data_window * window
    fft_result = np.fft.rfft(windowed_data, n=FFT_SIZE)
    mag = np.abs(fft_result) * 2.0 / BUFFER_SIZE
    
    current_magnitude[:] = mag[:]
    
    if resp_frecuencia_on:
        max_magnitudes_visuales[:] = np.maximum(max_magnitudes_visuales, mag)

# --- Interfaz Gráfica ---
fig = plt.figure(figsize=(10, 8.5))
fig.canvas.manager.set_window_title('UPCT - TFG Analizador de Audio (Fourier)')

# 1. Señal Temporal
gs_time = fig.add_gridspec(1, 1, top=0.94, bottom=0.78, left=0.08, right=0.95)
ax_time = fig.add_subplot(gs_time[0])
times = np.linspace(0, BUFFER_SIZE/FS, BUFFER_SIZE)
line_time, = ax_time.plot(times, np.zeros(BUFFER_SIZE), color='#00ffcc', lw=0.8)

ax_time.set_ylim(-max_time_y, max_time_y) 
ax_time.set_xlim(0, max_time_x)
ax_time.set_title('Onda en Tiempo Real (Micrófono)', color='white', fontsize=12)
ax_time.set_ylabel('Amplitud', fontsize=9, color='grey')
ax_time.set_xlabel('Tiempo (s)', fontsize=9, color='grey')
ax_time.grid(True, color='#333333', alpha=0.5)

# 2. Espectro y Cascada
gs_freq = fig.add_gridspec(2, 1, top=0.70, bottom=0.15, left=0.08, right=0.95, hspace=0.0)

ax_spec = fig.add_subplot(gs_freq[0])
ax_waterfall = fig.add_subplot(gs_freq[1], sharex=ax_spec)

line_spec, = ax_spec.plot([], [], color='#FFD700', lw=1.2)
line_resp_freq, = ax_spec.plot([], [], color='red', lw=2.0, alpha=0.8)

peak_scatter = ax_spec.scatter([], [], color='#00FFFF', marker='o', s=40, zorder=10)
peak_texts = [] 

xticks_log = np.log10([20, 100, 1000, 10000, 20000])
xticklabels_log = ['20', '100', '1k', '10k', '20k']
minor_ticks = []
for decade in [10, 100, 1000, 10000]:
    for i in range(2, 10):
        val = decade * i
        if 20 <= val <= 20000:
            minor_ticks.append(val)
minor_xticks_log = np.log10(minor_ticks)

xticks_lin = [20, 5000, 10000, 15000, 20000]
xticklabels_lin = ['20', '5k', '10k', '15k', '20k']

# --- CONFIGURACIÓN INICIAL ---
ax_spec.set_xlim(F_MIN, current_f_max)
ax_spec.set_ylim(0, max_freq_y)  
ax_spec.set_ylabel('Magnitud (Lineal)', fontsize=9, color='grey')
ax_spec.set_xticks(xticks_lin)
ax_spec.set_xticks([], minor=True)
ax_spec.tick_params(labelbottom=False) 
ax_spec.grid(True, which='major', color='#666666', ls='-', alpha=0.8)
ax_spec.grid(True, which='minor', color='#333333', ls=':', alpha=0.6)

img_waterfall = ax_waterfall.imshow(
    waterfall_data, 
    aspect='auto', 
    cmap='magma', 
    vmin=-120, vmax=-40,
    extent=[F_MIN, current_f_max, waterfall_history, 0] 
)
ax_waterfall.set_xticks(xticks_lin)
ax_waterfall.set_xticklabels(xticklabels_lin, color='grey')
ax_waterfall.set_xticks([], minor=True)
ax_waterfall.set_xlabel('Frecuencia (Hz) - Escala Lineal', fontsize=10, color='grey')
ax_waterfall.set_yticks([]) 
ax_waterfall.grid(True, which='major', color='#666666', ls='-', alpha=0.8)
ax_waterfall.grid(True, which='minor', color='#333333', ls=':', alpha=0.6)


# =============================================================================
# --- CONTROLES INTERACTIVOS ---
# =============================================================================

# 0. BOTÓN: Estado RUN/HOLD 
ax_btn_freeze = fig.add_axes([0.02, 0.04, 0.08, 0.04]) 
btn_freeze = Button(ax_btn_freeze, 'Estado: RUN', color='#222222', hovercolor='#444444')
btn_freeze.label.set_color('#00ffcc') 
btn_freeze.label.set_fontweight('bold')
btn_freeze.label.set_fontsize(9)
for spine in ax_btn_freeze.spines.values(): spine.set_edgecolor('#555555'); spine.set_linewidth(1.0)

def toggle_freeze(event):
    global is_frozen, ani
    is_frozen = not is_frozen
    if is_frozen:
        btn_freeze.label.set_text('Estado: HOLD')
        btn_freeze.color = '#8B0000'           
        btn_freeze.label.set_color('white')  
        for spine in ax_btn_freeze.spines.values(): spine.set_edgecolor('#FF0000')
        ani.event_source.stop() 
    else:
        btn_freeze.label.set_text('Estado: RUN')
        btn_freeze.color = '#222222'           
        btn_freeze.label.set_color('#00ffcc')    
        for spine in ax_btn_freeze.spines.values(): spine.set_edgecolor('#555555')     
        ani.event_source.start() 
    fig.canvas.draw()
btn_freeze.on_clicked(toggle_freeze)

# 1. Ajuste Amplitud Tiempo (Eje Y Onda)
axbox_ty = fig.add_axes([0.15, 0.04, 0.04, 0.04])  
tb_time_y = TextBox(axbox_ty, 'A Onda: ', initial=str(max_time_y), color='#333333', hovercolor='#444444')
def submit_time_y(text):
    global max_time_y
    try:
        val = float(text)
        if val > 0:
            max_time_y = val
            ax_time.set_ylim(-val, val)
    except ValueError: pass
tb_time_y.on_submit(submit_time_y)

# 2. Ajuste Límite Tiempo (Eje X Onda)
axbox_tx = facility = fig.add_axes([0.27, 0.04, 0.04, 0.04])  
tb_time_x = TextBox(axbox_tx, 't Onda: ', initial=f"{max_time_x:.3f}", color='#333333', hovercolor='#444444')
def submit_time_x(text):
    global max_time_x
    try:
        val = float(text)
        if val > 0:
            max_time_x = val
            ax_time.set_xlim(0, val)
    except ValueError: pass
tb_time_x.on_submit(submit_time_x)

# 3. Ajuste Magnitud Espectro (Eje Y Frecuencia) 
axbox_fy = fig.add_axes([0.38, 0.04, 0.04, 0.04])  
tb_freq_y = TextBox(axbox_fy, 'M Espec: ', initial=str(max_freq_y), color='#333333', hovercolor='#444444')
def submit_freq_y(text):
    global max_freq_y, max_mag_db
    try:
        val = float(text)
        if mag_scale_mode == 'lin':
            if val > 0:
                max_freq_y = val
                ax_spec.set_ylim(0, max_freq_y) 
        else:
            max_mag_db = val
            ax_spec.set_ylim(-100, max_mag_db)
    except ValueError: pass
tb_freq_y.on_submit(submit_freq_y)

# 4. NUEVO Ajuste Límite Frecuencia Espectro (Eje X Frecuencia)
axbox_fx = fig.add_axes([0.50, 0.04, 0.04, 0.04])  
tb_freq_x = TextBox(axbox_fx, 'F Espec: ', initial=str(current_f_max), color='#333333', hovercolor='#444444')
def submit_freq_x(text):
    global current_f_max
    try:
        val = float(text)
        if val > F_MIN and val <= FS/2: # El límite máximo es la frecuencia de Nyquist
            current_f_max = val
            if scale_mode == 'lin':
                ax_spec.set_xlim(F_MIN, current_f_max)
                ax_waterfall.set_xlim(F_MIN, current_f_max)
                img_waterfall.set_extent([F_MIN, current_f_max, waterfall_history, 0])
            else:
                ax_spec.set_xlim(np.log10(F_MIN), np.log10(current_f_max))
                ax_waterfall.set_xlim(np.log10(F_MIN), np.log10(current_f_max))
                img_waterfall.set_extent([np.log10(F_MIN), np.log10(current_f_max), waterfall_history, 0])
            fig.canvas.draw()
    except ValueError: pass
tb_freq_x.on_submit(submit_freq_x)

# 5. Botón de Escala FRECUENCIA (Eje X)
ax_btn_scale = fig.add_axes([0.56, 0.04, 0.20, 0.04])
btn_scale = Button(ax_btn_scale, 'Escala Frec: LINEAL', color='#112222', hovercolor='#444444')
btn_scale.label.set_color('#00ffcc') 
btn_scale.label.set_fontweight('bold')

def toggle_scale(event):
    global scale_mode, waterfall_data
    if scale_mode == 'lin':
        scale_mode = 'log'
        btn_scale.label.set_text('Escala Frec: LOG')
        btn_scale.label.set_color('#FFD700') 
        btn_scale.color = '#222222'           
        
        ax_spec.set_xlim(np.log10(F_MIN), np.log10(current_f_max))
        ax_spec.set_xticks(xticks_log)
        ax_spec.set_xticks(minor_xticks_log, minor=True)
        
        ax_waterfall.set_xlim(np.log10(F_MIN), np.log10(current_f_max))
        ax_waterfall.set_xticks(xticks_log)
        ax_waterfall.set_xticklabels(xticklabels_log, color='grey')
        ax_waterfall.set_xticks(minor_xticks_log, minor=True)
        ax_waterfall.set_xlabel('Frecuencia (Hz) - Escala Logarítmica', fontsize=10, color='grey')
        img_waterfall.set_extent([np.log10(F_MIN), np.log10(current_f_max), waterfall_history, 0])
    else:
        scale_mode = 'lin'
        btn_scale.label.set_text('Escala Frec: LINEAL')
        btn_scale.label.set_color('#00ffcc') 
        btn_scale.color = '#112222'           
        
        ax_spec.set_xlim(F_MIN, current_f_max)
        ax_spec.set_xticks(xticks_lin)
        ax_spec.set_xticks([], minor=True)
        
        ax_waterfall.set_xlim(F_MIN, current_f_max)
        ax_waterfall.set_xticks(xticks_lin)
        ax_waterfall.set_xticklabels(xticklabels_lin, color='grey')
        ax_waterfall.set_xticks([], minor=True)
        ax_waterfall.set_xlabel('Frecuencia (Hz) - Escala Lineal', fontsize=10, color='grey')
        img_waterfall.set_extent([F_MIN, current_f_max, waterfall_history, 0])
        
    waterfall_data.fill(-120.0) 
    fig.canvas.draw() 
btn_scale.on_clicked(toggle_scale)

# 6. NUEVO BOTÓN: Escala MAGNITUD (Eje Y)
ax_btn_mag = fig.add_axes([0.77, 0.04, 0.20, 0.04])
btn_mag = Button(ax_btn_mag, 'Escala Mag: LINEAL', color='#112222', hovercolor='#444444')
btn_mag.label.set_color('#00ffcc') 
btn_mag.label.set_fontweight('bold')

def toggle_mag_scale(event):
    global mag_scale_mode
    if mag_scale_mode == 'lin':
        mag_scale_mode = 'log'
        btn_mag.label.set_text('Escala Mag: LOG (dB)')
        btn_mag.label.set_color('#FFD700')
        btn_mag.color = '#222222'
        
        ax_spec.set_ylabel('Magnitud (dB)', fontsize=9, color='grey')
        ax_spec.set_ylim(-100, max_mag_db)
        tb_freq_y.set_val(str(max_mag_db)) 
    else:
        mag_scale_mode = 'lin'
        btn_mag.label.set_text('Escala Mag: LINEAL')
        btn_mag.label.set_color('#00ffcc')
        btn_mag.color = '#112222'
        
        ax_spec.set_ylabel('Magnitud (Lineal)', fontsize=9, color='grey')
        ax_spec.set_ylim(0, max_freq_y)
        tb_freq_y.set_val(str(max_freq_y)) 
    fig.canvas.draw()
btn_mag.on_clicked(toggle_mag_scale)

# --- BOTONES FLOTANTES SUPERIORES ---
# Botón Respuesta en Frecuencia 
ax_btn = fig.add_axes([0.80, 0.64, 0.14, 0.04])
btn_toggle = Button(ax_btn, 'Resp. Frec: OFF', color='#222222', hovercolor='#444444')
btn_toggle.label.set_color('white')
btn_toggle.label.set_fontweight('bold')
btn_toggle.label.set_fontsize(9)
for spine in ax_btn.spines.values(): spine.set_edgecolor('#555555'); spine.set_linewidth(1.0)

def toggle_resp_frecuencia(event):
    global resp_frecuencia_on, max_magnitudes_visuales
    resp_frecuencia_on = not resp_frecuencia_on
    if resp_frecuencia_on:
        btn_toggle.label.set_text('● ON')
        btn_toggle.color = '#8B0000'           
        btn_toggle.label.set_color('#00FFFF')  
        for spine in ax_btn.spines.values(): spine.set_edgecolor('#FF0000')     
        max_magnitudes_visuales[:] = current_magnitude[:]
    else:
        btn_toggle.label.set_text('Resp. Frec: OFF')
        btn_toggle.color = '#222222'           
        btn_toggle.label.set_color('white')    
        for spine in ax_btn.spines.values(): spine.set_edgecolor('#555555')     
        line_resp_freq.set_data([], [])
    fig.canvas.draw()
btn_toggle.on_clicked(toggle_resp_frecuencia)

# Botón Trigger ON/OFF 
ax_btn_trig = fig.add_axes([0.80, 0.945, 0.14, 0.04])
btn_trig = Button(ax_btn_trig, 'Trigger: OFF', color='#222222', hovercolor='#444444')
btn_trig.label.set_color('white')
btn_trig.label.set_fontweight('bold')
btn_trig.label.set_fontsize(9)
for spine in ax_btn_trig.spines.values(): spine.set_edgecolor('#555555'); spine.set_linewidth(1.0)

def toggle_trigger(event):
    global trigger_on
    trigger_on = not trigger_on
    if trigger_on:
        btn_trig.label.set_text('Trigger: ON')
        btn_trig.color = '#005580'           
        btn_trig.label.set_color('white')  
        for spine in ax_btn_trig.spines.values(): spine.set_edgecolor('#00FFFF')
    else:
        btn_trig.label.set_text('Trigger: OFF')
        btn_trig.color = '#222222'           
        btn_trig.label.set_color('white')    
        for spine in ax_btn_trig.spines.values(): spine.set_edgecolor('#555555')     
    fig.canvas.draw() 
btn_trig.on_clicked(toggle_trigger)


# =============================================================================
# --- MOTOR DE ANIMACIÓN ---
# =============================================================================
def update_plots(frame):
    global pointer, waterfall_data, peak_texts
    
    display_samples = int(max_time_x * FS)
    search_margin = int(FS * 0.05) if trigger_on else 0 
    total_extract = display_samples + search_margin
    
    if total_extract > FFT_SIZE: total_extract = FFT_SIZE
        
    p = pointer % FFT_SIZE
    if p >= total_extract:
        raw_data = audio_buffer[p - total_extract : p]
    else:
        raw_data = np.concatenate((audio_buffer[-(total_extract - p):], audio_buffer[:p]))
    
    data = raw_data[-display_samples:] 
    
    if trigger_on:
        crossings = np.where((raw_data[:-1] <= 0) & (raw_data[1:] > 0))[0]
        valid_crossings = crossings[crossings <= len(raw_data) - display_samples]
        if len(valid_crossings) > 0:
            trigger_idx = valid_crossings[-1]
            data = raw_data[trigger_idx : trigger_idx + display_samples]
        
    mag = current_magnitude.copy()
    
    peak_indices, _ = find_peaks(mag, height=PEAK_HEIGHT_THRESHOLD, prominence=PEAK_PROMINENCE_THRESHOLD, distance=MIN_DISTANCE_BETWEEN_PEAKS)
    
    if len(peak_indices) > 0:
        sorted_peak_indices = np.array(peak_indices)[np.argsort(mag[peak_indices])[::-1]][:8]
        peak_freqs = freqs_global[sorted_peak_indices]
        peak_mags_base = mag[sorted_peak_indices]
    else:
        peak_freqs = []
        peak_mags_base = []
    
    valid_idx = freqs_global > 0
    freqs_valid = freqs_global[valid_idx]
    
    # --- CÁLCULO DE MAGNITUD (Lineal vs dB) ---
    mags_valid_base = mag[valid_idx]
    max_mags_visuales_base = max_magnitudes_visuales[valid_idx]
    
    if mag_scale_mode == 'log':
        mags_plot = 20 * np.log10(mags_valid_base + 1e-12)
        max_mags_plot = 20 * np.log10(max_mags_visuales_base + 1e-12)
        peak_mags_plot = 20 * np.log10(peak_mags_base + 1e-12) if len(peak_mags_base) > 0 else []
    else:
        mags_plot = mags_valid_base
        max_mags_plot = max_mags_visuales_base
        peak_mags_plot = peak_mags_base
        
    # --- CÁLCULO DE FRECUENCIA (Lineal vs Log) ---
    if scale_mode == 'log':
        x_plot = np.log10(freqs_valid)
        x_peaks = np.log10(peak_freqs) if len(peak_freqs) > 0 else []
    else:
        x_plot = freqs_valid
        x_peaks = peak_freqs
    
    # 1. Dibujar Temporal
    times = np.linspace(0, display_samples / FS, display_samples)
    line_time.set_data(times, data)
    
    # 2. Dibujar Espectro
    line_spec.set_data(x_plot, mags_plot)
    
    # Limpieza forzada de la línea de Respuesta en Frecuencia al desactivarla
    if resp_frecuencia_on:
        line_resp_freq.set_data(x_plot, max_mags_plot)
    else:
        line_resp_freq.set_data([], []) 
    
    if len(peak_freqs) > 0:
        peak_scatter.set_offsets(np.column_stack((x_peaks, peak_mags_plot)))
    else:
        peak_scatter.set_offsets(np.zeros((0, 2)))

    for text in peak_texts: text.remove()
    peak_texts.clear()
    
    y_range = ax_spec.get_ylim()[1] - ax_spec.get_ylim()[0]
    for i in range(len(peak_freqs)):
        if mag_scale_mode == 'log':
            label = f"({peak_freqs[i]:.0f} Hz, {peak_mags_plot[i]:.1f} dB)"
        else:
            label = f"({peak_freqs[i]:.0f} Hz, {peak_mags_plot[i]:.3f})"
            
        t = ax_spec.text(x_peaks[i], peak_mags_plot[i] + (y_range * 0.04), label, color='#00FFFF', fontsize=8, ha='center', weight='bold')
        peak_texts.append(t)
        
    # 3. Dibujar Cascada 
    magnitude_db = 20 * np.log10(mag + 1e-12)
    if scale_mode == 'log':
        mag_db_mapped = np.interp(log_freqs, freqs_global, magnitude_db)
    else:
        mag_db_mapped = np.interp(lin_freqs, freqs_global, magnitude_db)
        
    waterfall_data = np.roll(waterfall_data, 1, axis=0)
    waterfall_data[0, :] = mag_db_mapped
    img_waterfall.set_data(waterfall_data)
    
    return [line_time, line_spec, line_resp_freq, peak_scatter, img_waterfall] + peak_texts

# --- Ejecución ---
print(f"Iniciando analizador en dispositivo ID {selected_device_id}...")
stream = sd.InputStream(device=selected_device_id, callback=audio_callback, samplerate=FS)
stream.start()

global ani
ani = animation.FuncAnimation(fig, update_plots, interval=50, blit=False, cache_frame_data=False)
plt.show()

stream.stop()
stream.close()
print("Analizador cerrado.")