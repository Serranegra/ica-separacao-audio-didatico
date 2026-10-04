# ICA para separação de fontes de áudio

Este projeto é um exercício didático de separação cega de fontes usando Independent Component Analysis (ICA), com foco em um caso simples de duas fontes de áudio misturadas linearmente.

O objetivo é demonstrar, na prática, como algoritmos como `FastICA` podem recuperar sinais originais a partir de misturas observadas.

## O que o código faz

O projeto possui dois exemplos principais:

- `example/example.py`: demonstra a ideia em um cenário controlado, usando arquivos de áudio já disponíveis localmente
- `from-youtube/ICA_youtube.py`: baixa dois vídeos do YouTube, extrai os áudios, mistura as fontes, recorta trechos escolhidos e aplica ICA para separar.

Em ambos os cenários, o fluxo é o mesmo:

1. Carrega duas fontes de áudio originais
2. Cria uma mistura linear instantânea entre elas
3. Salva a mistura em arquivos `.wav`
4. Aplica `FastICA` para estimar as fontes separadas
5. Normaliza os sinais e salva os resultados em arquivos de saída

## O que é ICA:

ICA (Independent Component Analysis ou Análise de Componentes Independentes) é uma técnica usada para separar sinais misturados quando não sabemos a mistura exata e apenas observamos as combinações resultantes deles.

A ideia central é:

- as fontes originais são assumidas como estatisticamente independentes
- as observações são combinações lineares dessas fontes
- o algoritmo tenta encontrar uma transformação que revele as fontes independentes

Matematicamente, a mistura pode ser escrita como:

`X = S * A^T`

onde:

- `S` é a matriz das fontes originais
- `A` é a matriz de mistura
- `X` é a matriz das observações (misturas)

O objetivo da ICA é estimar uma matriz de separação `W` de forma que:

`S_est = X * W`

ou seja, recuperar as fontes que geraram as misturas observadas.

## Como isso funciona neste projeto

No script principal:

```python
S = np.c_[a1, a2]
A = np.array([
    [0.7, 0.3],
    [0.2, 0.8]
])
X = np.dot(S, A.T)
```

Aqui:

- `a1` e `a2` são as duas fontes de áudio
- `S` empilha os dois sinais em uma matriz
- `A` representa como cada fonte contribui para cada microfone/mistura
- `X` são as misturas observadas

Depois disso, o código faz:

```python
ica = FastICA(n_components=2, random_state=42)
S_recuperado = ica.fit_transform(X)
```

Esse passo aplica o algoritmo `FastICA`, que tenta recuperar duas fontes independentes a partir da mistura. O resultado são sinais separados que, em um caso ideal, aproximam as fontes originais.

A normalização final garante que os sinais separados tenham amplitude adequada para gravação em `.wav`.

## Por que isso não funciona no mundo real:

Este projeto não tenta reproduzir um ambiente de gravação real. Ele usa um cenário idealizado, com apenas duas fontes, mistura linear instantânea, ausência de reverberação, ausência de atraso entre os sinais, etc. Isso torna o problema bem-sucedido para fins de aprendizado, mas também mostra por que o método é frágil quando levado ao mundo real. Algumas **limitações** conhecidas (mas não as únicas) são:

### 1. Dependência estatística entre as fontes

A teoria da ICA assume que as fontes são independentes. Em áudio real, isso nem sempre é verdade:

- vozes humanas podem ter estrutura temporal e harmônicos correlacionados
- música e ruído podem não ser completamente independentes
- sinais podem compartilhar padrões temporais ou frequenciais

### 2. Mistura não linear ou com reverberação

Em um ambiente real, o som não chega a cada microfone por uma simples combinação linear de amplitudes. O som pode refletir em paredes e objetos, chegar com atraso e até ser filtrado pelo ambiente. Violando a hipótese da ICA, que assume mistura instantânea e linear.

### 3. Microfones reais introduzem ruído e diferenças físicas

Diferentes microfones têm respostas de frequência diferentes. Essas diferenças tornam a mistura menos “limpa” e podem causar separação ruim ou instável.

## Estrutura do projeto

```text
ICA/
├── README.md
├── example/
│   ├── example.py
│   ├── fontes/
│   ├── mistura/
│   └── resultado/
├── from-youtube/
    └── ICA_youtube.py
```

## Dependências

O projeto usa:

- Python
- NumPy
- librosa
- soundfile
- scikit-learn
- yt-dlp
- imageio-ffmpeg

## Como executar

### Exemplo sintético

Execute:

```bash
python example/example.py
```

### Exemplo com vídeos do YouTube

Edite o arquivo `from-youtube/ICA_youtube.py` e substitua os links:

```python
url_video1 = 'LINK DO VIDEO 1 AQUI'
url_video2 = 'LINK DO VIDEO 2 AQUI'
```

Depois rode:

```bash
python from-youtube/ICA_youtube.py
```

## Conclusão

Espero mostrar um uso da ICA em cenário simplificado, mas também deixar claro o motivo de sua fragilidade em aplicações reais: a teoria assume condições muito mais limpas do que as encontradas em gravações do mundo real.

## Referências

Para um tratamento matemática por trás do algoritmo, recomendo a aula clássica de **Independent Component Analysis (ICA)** do curso de Machine Learning de Stanford (**CS229**), ministrada pelo **Andrew Ng**: [[Andrew Ng - Independent Component Analysis (YouTube)](https://youtu.be/dyb_cFywuik?si=ipiLhQZdTV_o5Tjq)]
