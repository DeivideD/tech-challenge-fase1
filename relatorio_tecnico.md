# Relatório Técnico — Tech Challenge Fase 1
## Sistema de Apoio ao Diagnóstico de Câncer de Mama

---

## 1. Introdução e definição do problema

Redes de saúde especializadas no atendimento à mulher lidam com um volume crescente de pacientes e com a necessidade de identificar rapidamente situações de risco. Atrasos ou erros na triagem de exames podem ter consequências graves.

Este projeto constrói a base de um sistema de **Machine Learning** capaz de classificar tumores de mama como **malignos** ou **benignos** a partir de medidas extraídas de exames de imagem. O objetivo é oferecer uma **ferramenta de apoio à decisão** que acelere a triagem e destaque casos de risco — sempre preservando o julgamento final do médico.

O problema é de **classificação binária supervisionada**: a partir de 30 medidas numéricas, prever uma de duas classes (maligno ou benigno).

---

## 2. A base de dados

Foi utilizada a base **Breast Cancer Wisconsin (Diagnostic)**, pública e amplamente validada na literatura, disponível na biblioteca scikit-learn.

- **569 pacientes** (amostras de tumores)
- **30 medidas** por paciente, derivadas da imagem digitalizada de uma punção aspirativa por agulha fina (PAAF). As medidas descrevem o núcleo das células (raio, textura, perímetro, área, suavidade, compacidade, concavidade, pontos côncavos, simetria e dimensão fractal), cada uma em três versões: média (*mean*), erro padrão (*error*) e pior caso (*worst*).
- **Alvo:** 0 = Maligno · 1 = Benigno

A escolha desta base se justifica por ser confiável, bem documentada e diretamente alinhada à tarefa principal sugerida no desafio (diagnóstico de câncer de mama).

---

## 3. Análise exploratória dos dados (EDA)

### 3.1 Distribuição das classes

A base contém **212 casos malignos (37,3%)** e **357 benignos (62,7%)**, caracterizando um **desbalanceamento moderado**.

Esse desbalanceamento tem uma implicação crítica: um modelo ingênuo que classificasse todas as pacientes como "benigno" alcançaria 62,7% de acurácia, mas deixaria passar **100% dos cânceres**. Essa constatação norteou toda a escolha de métricas de avaliação (Seção 6).

### 3.2 Qualidade dos dados

A verificação de valores ausentes não identificou nenhum dado faltante (0 ocorrências). Portanto, **não foi necessária imputação**. A ausência de inconsistências permitiu focar o pré-processamento na preparação para a modelagem.

### 3.3 Padrões identificados

A comparação das médias por classe revelou um padrão consistente: **tumores malignos apresentam valores maiores em praticamente todas as medidas**, em especial nas relacionadas a tamanho (raio, perímetro, área) e à irregularidade do contorno (concavidade e pontos côncavos).

| Medida | Maligno (média) | Benigno (média) |
|---|---|---|
| mean radius | 17,46 | 12,15 |
| mean perimeter | 115,37 | 78,08 |
| mean area | 978,38 | 462,79 |
| mean concavity | 0,16 | 0,05 |
| mean concave points | 0,09 | 0,03 |
| worst area | 1422,29 | 558,90 |

Os histogramas confirmaram visualmente esse padrão: as medidas `mean concave points` e `mean concavity` mostraram a melhor separação entre as classes, com sobreposição mínima — indicando alto poder discriminativo.

### 3.4 Análise de correlação

A correlação entre cada medida e o diagnóstico confirmou que as medidas de **tamanho** e de **pontos côncavos** são as mais associadas à malignidade. Em contraste, medidas como `fractal dimension error`, `texture error` e `symmetry error` apresentaram correlação próxima de zero.

A análise de correlação entre as próprias medidas revelou forte **multicolinearidade** no bloco de tamanho: `mean radius`, `mean perimeter` e `mean area` têm correlação ≈ 0,99 entre si — esperado, pois são formas geométricas de medir a mesma característica. Já a concavidade mostrou-se relativamente independente do tamanho, carregando informação complementar. Essa observação ajuda a justificar o teste de múltiplos modelos, já que a multicolinearidade afeta modelos lineares de forma diferente dos baseados em árvores.

---

## 4. Pré-processamento

O pipeline de pré-processamento seguiu três etapas:

1. **Separação de features (X) e alvo (y):** as 30 medidas como variáveis preditoras e o diagnóstico como variável-alvo.

2. **Divisão treino/teste:** separação estratificada de 80% para treino (455 pacientes) e 20% para teste (114 pacientes). A estratificação (`stratify=y`) preservou a proporção 37/63 nos dois conjuntos, e a semente fixa (`random_state=42`) garante reprodutibilidade.

3. **Padronização:** as medidas possuem escalas muito distintas (área na casa dos milhares; suavidade na casa dos centésimos). Aplicou-se `StandardScaler` para colocar todas as medidas na mesma escala (média 0, desvio 1).

> **Cuidado contra *data leakage*:** o padronizador foi ajustado (*fit*) **apenas** nos dados de treino e depois aplicado (*transform*) no teste. Isso evita que informações do conjunto de teste influenciem o treinamento, garantindo uma avaliação honesta.

---

## 5. Modelagem

Foram treinados e comparados **três modelos** de naturezas diferentes:

- **Regressão Logística** — modelo linear, interpretável, eficaz quando a separação entre classes é relativamente linear.
- **Árvore de Decisão** — modelo baseado em regras, altamente interpretável (`max_depth=4` para evitar sobreajuste).
- **Random Forest** — conjunto (*ensemble*) de 200 árvores que votam, robusto e excelente para estimar a importância das variáveis.

A diversidade de abordagens permite uma comparação justa e fundamenta a escolha do modelo final com base em evidência.

---

## 6. Avaliação

### 6.1 Escolha da métrica

Em um problema de diagnóstico de câncer, **os dois tipos de erro não têm o mesmo peso**:

- **Falso negativo** (classificar um tumor maligno como benigno): gravíssimo — a paciente deixa de receber tratamento.
- **Falso positivo** (classificar um tumor benigno como maligno): gera ansiedade e exames adicionais, mas é corrigível por exames subsequentes.

Por isso, embora a acurácia seja reportada, a **métrica prioritária é o recall da classe maligna** (sensibilidade): a proporção dos cânceres reais que o modelo consegue detectar. Minimizar falsos negativos é o objetivo central de um sistema de triagem.

### 6.2 Resultados

| Modelo | Acurácia | Recall (maligno) | Precisão (maligno) | F1 (maligno) |
|---|---|---|---|---|
| **Regressão Logística** ⭐ | **98,2%** | **97,6%** | **97,6%** | **97,6%** |
| Árvore de Decisão | 93,9% | 92,9% | 90,7% | 91,8% |
| Random Forest | 95,6% | 92,9% | 95,1% | 94,0% |

A **Regressão Logística** apresentou o melhor desempenho em todas as métricas. Sua matriz de confusão revelou apenas **1 falso negativo** e **1 falso positivo** entre as 114 pacientes do teste.

### 6.3 Discussão: por que o modelo mais simples venceu

Um resultado contraintuitivo, mas valioso: o Random Forest, mais complexo, **não** superou a Regressão Logística. Isso ocorre porque a relação entre as medidas e o diagnóstico nesta base é bem comportada e aproximadamente linear (como evidenciado pela boa separação das classes na EDA). Nesse cenário, um modelo linear captura o padrão de forma eficaz, sem a complexidade adicional do *ensemble*. O resultado reforça um princípio importante: **o modelo mais simples que resolve bem o problema é, em geral, a melhor escolha** (princípio da parcimônia).

---

## 7. Explicabilidade

Para que um sistema de IA seja confiável em medicina, suas decisões precisam ser auditáveis — não uma "caixa-preta".

### 7.1 Feature importance (Random Forest)

As cinco medidas mais influentes foram: `worst perimeter`, `worst area`, `worst concave points`, `mean concave points` e `worst radius`. Ou seja, **tamanho e irregularidade do contorno** — exatamente os padrões identificados na análise exploratória.

### 7.2 Análise SHAP

A análise SHAP confirmou, paciente a paciente, a coerência do modelo: **valores altos** de tamanho e de pontos côncavos empurram a decisão para "maligno", enquanto **valores baixos** empurram para "benigno". Esse comportamento é clinicamente coerente e aumenta a confiabilidade do modelo como ferramenta de apoio.

### 7.3 Convergência das evidências

O mesmo conjunto de variáveis (tamanho e pontos côncavos) emergiu como mais relevante por **cinco caminhos independentes**: comparação de médias, histogramas, correlação, primeira divisão da árvore de decisão e importância no Random Forest. Essa convergência confere robustez à conclusão de que essas características são os indicadores mais confiáveis de malignidade nesta base.

---

## 8. Conclusão e considerações sobre uso prático

O modelo final (Regressão Logística) atinge 98,2% de acurácia e 97,6% de recall na classe maligna, demonstrando viabilidade técnica como ferramenta de triagem.

**O modelo pode ser usado na prática?** Sim, **como apoio à decisão** — não como substituto do médico. Pontos a considerar:

- Mesmo com 97,6% de recall, o modelo ainda erra. Em saúde, cada erro tem impacto humano real, o que torna a supervisão médica indispensável.
- O modelo identifica **correlações**, não causas. Ele aponta quais medidas acompanham o diagnóstico, sem explicar mecanismos biológicos.
- O desempenho depende da qualidade dos dados de entrada. Uma base limpa e equilibrada favoreceu os resultados; em produção, dados ruidosos exigiriam validação contínua.

**Uso recomendado:** triagem inicial e priorização de casos, sinalizando exames de maior risco para atenção médica imediata, com o diagnóstico final sempre validado por um profissional.

### Trabalhos futuros
- Validação com dados de outras instituições para testar generalização.
- Extensão com Visão Computacional (CNN) para análise direta de mamografias.
- Interface interativa para uso por profissionais de saúde.

---

## 9. Reprodutibilidade

Todo o desenvolvimento está no notebook `notebooks/tech_challenge.ipynb`, executável de ponta a ponta (**Run → Run All Cells**). As dependências estão em `requirements.txt` e o modelo treinado em `src/`.
