# DCGAN para Geração de Rostos de Anime de Alta Fidelidade

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-%23F9AB00.svg?style=flat&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

Este repositório contém a implementação de uma Rede Neural Adversarial Convolucional Profunda (**DCGAN**) customizada e de alta capacidade para a geração de imagens sintéticas de rostos de anime na resolução $64 \times 64$ pixels.

## 📌 Premissa Fundamental

O objetivo central deste projeto é **construir um modelo generativo estável, puramente convolucional e de alta fidelidade**, superando as falhas clássicas de borrões estruturais e colapso de modo (*Mode Collapse*) através de um equilíbrio matemático rigoroso e assimétrico entre as capacidades do Gerador e do Discriminador.

---

## 🛠️ Pilares de Engenharia de Machine Learning

O pipeline foi projetado sob três pilares práticos para otimizar o aprendizado no ambiente Google Colab:

* **Otimização de Infraestrutura e I/O**: Extração direta do dataset compactado do Google Drive para o SSD NVMe local da máquina virtual (`/content/`), eliminando gargalos de leitura de disco e acelerando o tempo de época na GPU T4.
* **Alta Capacidade Geométrica (`ch_base = 128`)**: Alargamento dos canais convolucionais internos e remoção completa de camadas lineares (`nn.Linear`) iniciais. O ruído entra direto como um mapa de recursos quadridimensional, permitindo que a rede retenha traços complexos (olhos, contornos e cabelos nítidos).
* **Estabilidade Adversarial Ativa (`AdamW` + Taxas Assimétricas)**: Uso do otimizador `AdamW` com decaimento de peso (*weight decay*) para mitigar artefatos xadrez repetitivos e calibração fina das taxas de aprendizado (mantendo o Discriminador 4 vezes mais lento que o Gerador).

---

## 📐 Especificações da Arquitetura

O modelo foi projetado com uma distribuição de filtros dinâmica baseada no hiperparâmetro `ch_base = 128`.

### Fluxo do Gerador
Expande progressivamente o vetor latente $z \in \mathbb{R}^{100 \times 1 \times 1}$ utilizando Convoluções Transpostas (`nn.ConvTranspose2d`), normalização por lote (`nn.BatchNorm2d`) e ativações `ReLU`. A camada final aplica `Tanh` para mapear os pixels para o intervalo $[-1.0, 1.0]$.

$$\text{Vetor Latente (100, 1, 1)} \longrightarrow 4\times4\ (1024\ ch) \longrightarrow 8\times8\ (512\ ch) \longrightarrow 16\times16\ (256\ ch) \longrightarrow 32\times32\ (128\ ch) \longrightarrow 64\times64\ (3\ ch)$$

### Fluxo do Discriminador
Reduz a imagem real ou sintética de $64\times64 \times 3$ por meio de Convoluções Estruturadas (`nn.Conv2d` com `stride=2`), `BatchNorm2d` e ativações `LeakyReLU(0.2)`. A classificação final é puramente convolucional (`kernel_size=4, stride=1, padding=0`), achatada por um `Flatten` e mapeada por uma `Sigmoid` para retornar a probabilidade escalar.

---

## ⚙️ Hiperparâmetros de Treinamento

| Hiperparâmetro | Valor Configurado | Justificativa Técnica |
| :--- | :--- | :--- |
| **Dimensão Latente (`latent_dim`)** | `100` | Espaço vetorial para sementes de características variadas. |
| **Resolução da Imagem** | `(3, 64, 64)` | Padrão RGB quadrado estável para redes convolucionais DCGAN. |
| **Tamanho do Lote (`batch_size`)** | `64` | Equilíbrio entre estabilidade do gradiente e uso de memória da GPU. |
| **Canais Base (`ch_base`)** | `128` | Filtros dobrados para evitar borrões e enriquecer texturas de rostos. |
| **Otimizador** | `AdamW` | Desacopla o decaimento de peso (`1e-5`), eliminando artefatos repetitivos. |
| **Taxa de Aprendizado do Gerador** | `0.0002` | Ritmo padrão estável com momentum `betas=(0.5, 0.999)`. |
| **Taxa de Aprendizado do Discriminador** | `0.00005` | Reduzida para 1/4 da taxa do Gerador para evitar dominância precoce. |

---

## 🔬 Validação Científica (Espaço Latente)

Para comprovar que o modelo de fato aprendeu a topologia de distribuição dos rostos e não apenas memorizou o dataset, o código inclui um teste de **Interpolação Linear Contínua**.

Dada a amostragem de dois vetores distintos $z_1$ e $z_2$ no hiperespaço quadridimensional, calculamos caminhos intermediários ponderados por um coeficiente $\alpha \in [0, 1]$:

$$z_{interp} = (1 - \alpha) \cdot z_1 + \alpha \cdot z_2$$

Se a transição visual exibida na tela for contínua e suave (com feições, cores de cabelo e expressões se fundindo gradualmente frame a frame), fica provado matematicamente que a GAN mapeou um espaço latente robusto, contínuo e bem estruturado.

---

## 📂 Como Estruturar o Repositório

```text
├── .github/
├── data/                  # Pasta criada localmente na execução para o dataset
├── images_dgan/           # Diretório onde as grades visuais por época são salvas
├── main.ipynb             # Notebook contendo o pipeline completo executado no Colab
└── README.md              # Documentação do projeto
