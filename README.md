# Tech Challenge – Fase 1: Sistema de Apoio ao Diagnóstico de Câncer de Mama

Solução de Machine Learning para apoio ao diagnóstico e detecção de riscos relacionados à saúde da mulher, desenvolvida como parte do Tech Challenge (Fase 1) da Pós-Tech FIAP.

## 1. Sobre o projeto

Uma rede de hospitais especializados no atendimento à mulher busca um sistema inteligente que ajude profissionais de saúde a identificar precocemente condições de risco. Nesta primeira fase, construímos a base do sistema com foco em **Machine Learning**, capaz de classificar tumores de mama como **malignos** ou **benignos** a partir de medidas extraídas de exames.

O sistema foi pensado como uma **ferramenta de apoio à decisão** — a palavra final é sempre do médico.

## 2. Base de dados

Utilizamos o **Breast Cancer Wisconsin (Diagnostic)**, disponível publicamente e incluído na biblioteca scikit-learn.

- **569 pacientes** (amostras de tumores)
- **30 medidas** (features) por paciente, extraídas da imagem digitalizada de uma punção do tumor (raio, perímetro, área, concavidade, etc., nas versões média, erro e pior caso)
- **Alvo:** 0 = Maligno (212 casos, 37,3%) · 1 = Benigno (357 casos, 62,7%)
- **Sem valores ausentes**

Fonte: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data

## 3. Estrutura do repositório

```
tech-challenge-fase1/
├── README.md                  # este arquivo
├── requirements.txt           # bibliotecas necessárias
├── relatorio_tecnico.md       # relatório técnico completo
├── notebooks/
│   └── tech_challenge.ipynb   # notebook principal (todo o desenvolvimento)
├── src/
│   ├── modelo_diagnostico.pkl # modelo campeão treinado
│   └── scaler.pkl             # padronizador ajustado
├── data/                      # dados (carregados via scikit-learn)
└── results/
    └── figures/               # gráficos gerados
```

## 4. Como executar

### Pré-requisitos
- Python 3.10 ou superior

### Passo a passo

```bash
# 1. Clonar o repositório
git clone <git@github.com:DeivideD/tech-challenge-fase1.git>
cd tech-challenge-fase1

# 2. Criar e ativar um ambiente virtual
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Instalar as dependências
pip install -r requirements.txt

# 4. Abrir o notebook
jupyter notebook
```

No Jupyter, abra `notebooks/tech_challenge.ipynb` e execute as células de cima para baixo (menu **Run → Run All Cells**). O notebook é reprodutível: roda do início ao fim sem erros e produz os mesmos resultados.

## 5. Metodologia (resumo)

1. **Exploração dos dados** – análise de distribuições, estatísticas descritivas, correlação e identificação de padrões.
2. **Pré-processamento** – separação treino/teste estratificada (80/20) e padronização das medidas, sem vazamento de dados (*data leakage*).
3. **Modelagem** – três modelos treinados e comparados: Regressão Logística, Árvore de Decisão e Random Forest.
4. **Avaliação** – métricas accuracy, recall, precision e F1, com foco no **recall da classe maligna** (minimizar falsos negativos).
5. **Explicabilidade** – feature importance e SHAP para interpretar as decisões do modelo.

## 6. Resultados

| Modelo | Acurácia | Recall (maligno) | Precisão (maligno) | F1 (maligno) |
|---|---|---|---|---|
| **Regressão Logística** ⭐ | **98,2%** | **97,6%** | **97,6%** | **97,6%** |
| Árvore de Decisão | 93,9% | 92,9% | 90,7% | 91,8% |
| Random Forest | 95,6% | 92,9% | 95,1% | 94,0% |

**Modelo campeão: Regressão Logística** – melhor desempenho em todas as métricas, deixando passar apenas 1 caso maligno no conjunto de teste.

## 7. Tecnologias

Python · scikit-learn · pandas · numpy · matplotlib · seaborn · SHAP · Jupyter

## 8. Autores

> Preencha com os nomes e RMs dos integrantes do grupo.

- Anderson Alves de Oliveira Mares - RM376889
- Caio Leandro de Santana - RM377048
- Gabrielle Miguel Feitosa - RM376095
- Mariane Cristina de Oliveira Mendes - RM376673
- Duarte Deivide - RM376838

Link do GIT: https://github.com/DeivideD/tech-challenge-fase1.git

Link do vídeo de apresentação: https://vimeo.com/1222692298?share=copy&fl=sv&fe=ci

