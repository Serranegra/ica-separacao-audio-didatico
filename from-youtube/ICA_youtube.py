import numpy as np
import librosa
import soundfile as sf
from sklearn.decomposition import FastICA
import yt_dlp
import imageio_ffmpeg

# Obter o caminho do executável do ffmpeg baixado pelo pip
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

# ==========================================
# Configurações dos Vídeos e Recortes
# ==========================================
url_video1 = 'LINK DO VIDEO 1 AQUI'  # Substitua pelo link do primeiro vídeo
inicio1_min, inicio1_seg = 1, 0   # 1m00s

url_video2 = 'LINK DO VIDEO 2 AQUI'  # Substitua pelo link do segundo vídeo
inicio2_min, inicio2_seg = 0, 45  # 0m45s

duracao_desejada_seg = 30         # Janela de 30 segundos (ex: 1m a 1m30s)

# Convertendo para segundos totais
offset1 = inicio1_min * 60 + inicio1_seg
offset2 = inicio2_min * 60 + inicio2_seg

# ==========================================
# 1. Download dos Áudios do YouTube
# ==========================================
def baixar_audio_wav(url, prefixo):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{prefixo}.%(ext)s',
        'ffmpeg_location': FFMPEG_PATH,  # Usa o binário instalado pelo pip
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',     # Converte diretamente para .wav
            'preferredquality': '192',
        }],
        'quiet': True,
        'overwrites': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    return f'{prefixo}.wav'

# ==========================================
# 2. Carregar Apenas o Trecho Escolhido
# ==========================================
# offset define onde começa a leitura; duration define por quantos segundos ler
def load_audio_segments(file_v1, file_v2):
    a1, sr = librosa.load(file_v1, sr=None, offset=offset1, duration=duracao_desejada_seg)
    a2, _  = librosa.load(file_v2, sr=sr,   offset=offset2, duration=duracao_desejada_seg)

    # Trunca para garantir alinhamento exato de amostras se um arquivo for menor
    min_len = min(len(a1), len(a2))
    a1 = a1[:min_len]
    a2 = a2[:min_len]

    # Matriz de Fontes Independentes (S)
    S = np.c_[a1, a2]
    return S, sr
# ==========================================
# 3. Mistura Linear Instantânea (X = S * A^T)
# ==========================================
def mix_sources(S, sr):

    A = np.array([
        [0.7, 0.3],  # Mic 1
        [0.2, 0.8]   # Mic 2
    ])

    X = np.dot(S, A.T)

    sf.write('mistura/mistura_yt_mic2.wav', X[:, 1], sr)

    return X

# ==========================================
# 4. Executar o FastICA
# ==========================================
def run_fastica(X):
    ica = FastICA(n_components=2, random_state=42)
    S_recuperado = ica.fit_transform(X)
    return S_recuperado

# ==========================================
# 5. Normalizar e Salvar
# ==========================================
def save_separated_sources(S_recuperado, sr):
    f1 = S_recuperado[:, 0] / np.max(np.abs(S_recuperado[:, 0]))
    f2 = S_recuperado[:, 1] / np.max(np.abs(S_recuperado[:, 1]))

    sf.write('resultado/yt_separado_1.wav', f1, sr)
    sf.write('resultado/yt_separado_2.wav', f2, sr)

    print(f"Sucesso! Trecho de {len(f1)/sr:.1f}s processado a partir das posições escolhidas.")

def main():
    print("Baixando áudios do YouTube...")
    file_v1 = baixar_audio_wav(url_video1, 'downloads/yt_audio1')
    file_v2 = baixar_audio_wav(url_video2, 'downloads/yt_audio2')

    S, sr = load_audio_segments(file_v1, file_v2)
    X = mix_sources(S, sr)
    S_recuperado = run_fastica(X)
    save_separated_sources(S_recuperado, sr)

if __name__ == "__main__":
    main()