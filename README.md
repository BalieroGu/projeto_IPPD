# projeto_IPPD
Projeto de IPPD

# O problema: knn (nearest neighbours)

https://drive.google.com/file/d/1VuL0_x8Y3IoWzcIfp_B8sKBNQg-xLMR4/view?usp=sharing

# Relatório de Alterações e Estratégias de Paralelização no K-Nearest Neighbours (KNN)

## Objetivo

O objetivo do trabalho foi otimizar um código que implementa o algoritmo K-Nearest Neighbours (KNN) para classificação de pontos. 
O código original tinha execução sequencial e apresentava um desempenho limitado ao processar grandes quantidades de dados. 
Nosso foco foi paralelizar as operações computacionalmente intensivas, utilizando OpenMP, para alcançar maior eficiência e 
aproveitamento do hardware disponível.

---

## Código Original

O código original seguia as etapas clássicas do KNN:

1. **Entrada e Parsing**: Leitura de grupos de pontos com rótulos e da entrada do ponto a ser avaliado.
2. **Cálculo de Distâncias**: Para cada ponto nos grupos, calculava-se a distância Euclidiana (sem a raiz quadrada) em relação ao ponto-alvo.
3. **Seleção dos `k` Mais Próximos**: Ordenava-se as distâncias para determinar os `k` pontos mais próximos.
4. **Classificação**: Determinação do rótulo mais frequente entre os `k` pontos.

Embora funcional, o código apresentava gargalos significativos:

- O cálculo das distâncias era processado de forma sequencial, o que escalava mal para grandes conjuntos de dados.
- A seleção dos `k` menores distâncias envolvia múltiplas comparações em loops, aumentando o custo computacional.

---

## Alterações Implementadas

Realizamos diversas alterações no código, com foco em paralelismo e otimização:

### Paralelismo com OpenMP:

- Utilizamos `#pragma omp parallel` para criar uma região paralela, permitindo que múltiplos threads trabalhem simultaneamente.
- O cálculo das distâncias foi dividido entre threads com `#pragma omp for schedule(dynamic)`. 
  O agendamento dinâmico foi escolhido para balancear a carga de trabalho, especialmente considerando grupos de tamanhos diferentes.
- Resultados locais (distâncias e rótulos mais próximos por thread) foram combinados de forma eficiente usando `#pragma omp critical`.

### Uso de Heap para Seleção dos `k` Mais Próximos:

- Substituímos a abordagem sequencial de ordenação por um **heap mínimo** para armazenar as `k` menores distâncias.
- Essa estrutura garante inserção e atualização eficientes (**O(log k)**) em vez do custo **O(k)** de deslocamentos no array.

### Evitação de Redundâncias:

- Em vez de calcular todas as distâncias em um vetor global e ordená-lo, cada thread manteve seu próprio heap local de distâncias.
- Apenas as distâncias relevantes foram combinadas na fase de redução.

### Vetorização Automática:

- Garantimos que o cálculo das distâncias fosse estruturado de forma que o compilador pudesse aplicar instruções SIMD, 
  otimizando operações matemáticas repetitivas.

---

## Raciocínio da Paralelização do KNN

A paralelização foi planejada com base na estrutura natural do problema:

1. **Independência das Distâncias**: O cálculo de cada distância é independente, tornando o problema altamente adequado para paralelismo. 
   Dividimos a carga entre threads no nível dos grupos e dos pontos.
2. **Fusão de Resultados**: Após o cálculo das distâncias por thread, os resultados foram combinados de forma sincronizada. 
   Utilizamos `#pragma omp critical` para proteger a fusão e garantir a integridade dos dados globais.
3. **Redução de Sobrecarga**: Para minimizar a competição entre threads, cada thread usou vetores locais para seus resultados, 
   reduzindo acessos simultâneos a recursos compartilhados.

---

## Impacto das Alterações

### Melhoria de Performance:

- A divisão de trabalho entre threads permitiu uma aceleração significativa em CPUs multi-core.
- O uso de `schedule(dynamic)` otimizou o balanceamento da carga, especialmente em cenários com grupos de tamanhos variados.

### Escalabilidade:

- O algoritmo agora escala melhor com o aumento do número de grupos e pontos, aproveitando ao máximo os núcleos da CPU.

### Complexidade Reduzida:

- A substituição de ordenação por heap reduziu a complexidade do passo de seleção dos `k` menores pontos, um dos gargalos do código original.

### Facilidade de Uso:

- As alterações foram projetadas para serem transparentes em relação à entrada e saída do programa, mantendo a funcionalidade original.

---

Se houver perguntas específicas ou a necessidade de ajustes futuros, estamos à disposição para discutir!  
