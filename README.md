# hiaac-m4-experiments

Experimentos realizados pela meta 4 do HIAAC.

## Estrutura dos subdiretórios

Os experimentos são divididos em três subdiretórios principais, dependendo do seu estágio de desenvolvimento:

1. `preliminary_analysis`: contém códigos para análise dos conjuntos de dados e não paa execução de experimentos em si. São usados para entender melhor os dados que serão utilizados nos experimentos. Aqui incluem códigos para geração das *views* dos dados, análise de sinais, etc. Os subdiretórios são organizados por tópicos e contém os códigos e os relatórios das análises.

2. `experiments`: contém os experimentos propriamente ditos. Cada experimento é um subdiretório, que contém os códigos e os resultados do experimento.

3. `analysis`: contém os códigos e os relatórios das análises dos resultados dos experimentos. Cada subdiretório é relativo a um experimento listado na pasta `experiments` (com o mesmo nome), e contém os códigos e os relatórios das análises dos resultados do experimento.

---
**NOTA**

Todo subdiretório deve conter um arquivo `README.md` explicando o que contém o subdiretório e como utilizá-lo. Além disso, devem possuir um arquivo `requirements.txt` com as dependências do subdiretório.

---

Um exemplo da estrutura dos subdiretórios é mostrada abaixo:

```
hiaac-m4-experiments/
|- preliminary_analysis/
|   |- dataset_processing/
|   |   |- README.md
|   |   |- requirements.txt
|   |   |- dataset_analysis.ipynb
|   |- ...
|   |
|- experiments/
|   |- experiment_1/
|   |   |- README.md
|   |   |- requirements.txt
|   |   |- experiment.py
|   |   |- resultados/
|   |   |   |- execucao_1/
|   |   |   |- ...
|   |- ...
|   |
|- analysis/
|   |- experiment_1/
|   |   |- README.md
|   |   |- requirements.txt
|   |   |- analysis.ipynb
|   |- ...

```

Mais informações podem ser encontradas nos subdiretórios.