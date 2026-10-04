
import numpy as np
import librosa
import soundfile as sf
from sklearn.decomposition import FastICA

# ==========================================
# 2. Carregar e Ajustar os Dois Áudios
# ==========================================
# Carrega a melodia e a música de Bach, garantindo que ambos tenham a mesma taxa de amostragem

melodia, sr = librosa.load('audios/beep_box/song.wav', sr=None)
bach, _ = librosa.load('audios/beep_box/bach.mp3', sr=sr) 
tamanho_maximo = min(len(melodia), len(bach))

# Se quiser limitar estritamente a 20 segundos:
limite_20s = int(sr * 20)
tamanho_final = min(tamanho_maximo, limite_20s)

# Corta os dois áudios exatamente no mesmo tamanho a partir do segundo 0
melodia = melodia[:tamanho_final]
bach = bach[:tamanho_final]

# 3. Montar a matriz S (agora ambos começam no tempo 0,0s e terminam juntos)
S = np.c_[melodia, bach]

# 4. Matriz de Mistura Linear
A = np.array([
    [0.7, 0.3],
    [0.3, 0.7]
])

# Aplica a mistura simultânea
X = np.dot(S, A.T)

# Salva as duas misturas (ambas terão a melodia e o Bach tocando ao mesmo tempo)
sf.write('audios/beep_box/mistura_mic1.wav', X[:, 0], sr)
sf.write('audios/beep_box/mistura_mic2.wav', X[:, 1], sr)

# ==========================================
# 4. Executar o FastICA
# ==========================================
ica = FastICA(n_components=2, random_state=42)
S_recuperado = ica.fit_transform(X)

# ==========================================
# 5. Normalizar e Salvar
# ==========================================
f1 = S_recuperado[:, 0] / np.max(np.abs(S_recuperado[:, 0]))
f2 = S_recuperado[:, 1] / np.max(np.abs(S_recuperado[:, 1]))

sf.write('example/resultado/resultado_saida_1.wav', f1, sr)
sf.write('example/resultado/resultado_saida_2.wav', f2, sr)

print("Processo concluído com sucesso!")
# %%
