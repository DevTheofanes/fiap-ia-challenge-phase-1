# Tech Challenge - Fase 4

## Tech Challenge

Tech Challenge é o projeto que engloba os conhecimentos obtidos em todas as disciplinas dessa fase. Esta é uma atividade que, em princípio, deve ser desenvolvida em grupo. É importante atentar-se ao prazo de entrega, uma vez que essa atividade é obrigatória, valendo 90% da nota de todas as disciplinas da fase.

## Desafio

Com a IA integrada aos processos médicos especializados em saúde da mulher, analisando exames, documentos e apoiando decisões clínicas através do assistente especializado, a rede hospitalar agora deseja monitorar continuamente as pacientes por meio de dados multimodais (áudio, vídeo e texto) para identificar sinais precoces de risco específicos da saúde e segurança feminina.

Além de processar laudos e exames médicos, o sistema poderá ter em suas funcionalidades (escolha pelo menos duas opções):

- Analisar vídeos de partos, cirurgias ginecológicas, sessões de fisioterapia pós-parto e consultas para identificar padrões anômalos ou sinais de desconforto.
- Processar gravações de voz de pacientes em consultas, detectando sintomas relacionados à fala (depressão pós-parto, ansiedade, sinais de violência doméstica, fadiga hormonal).
- Detectar anomalias em sinais vitais específicos (pressão arterial em gestantes, batimentos fetais), prescrições hormonais e evolução clínica, alertando a equipe especializada em tempo real.
- Integrar tudo isso com serviços gerenciados em nuvem, como Azure Cognitive Services, mantendo os mais altos padrões de privacidade e segurança para dados sensíveis.

## Objetivo

Realizar a análise e fusão de diferentes tipos de dados médicos específicos da saúde da mulher — incluindo texto, áudio e vídeo. Sendo assim, escolha pelo menos três das opções abaixo:

- Detectar precocemente riscos em saúde materna e ginecológica.
- Identificar sinais de violência doméstica ou abuso.
- Monitorar bem-estar psicológico feminino.
- Utilizar serviços em nuvem para ampliar capacidade de processamento especializado.
- Aplicar técnicas de detecção de anomalias em tempo real para monitoramento preventivo específico.

## Requisitos Obrigatórios

### Entregas técnicas

#### 1. Análise de Vídeo Especializada para Saúde da Mulher

Processar vídeos clínicos específicos:

- Cirurgias: detecção de complicações ou sangramento anômalo.
- Consultas: identificação de sinais não-verbais de desconforto ou medo.
- Fisioterapia: análise de movimentos e recuperação.
- Triagem de violência: detecção de linguagem corporal indicativa de abuso.

Modelos especializados:

YOLOv8 customizado para detecção de um dos itens abaixo. Você pode escolher uma das opções listadas ou propor uma de sua parte.

- Instrumentos cirúrgicos ginecológicos.
- Áreas críticas em cirurgias (útero, ovários, mamas).
- Sangramento anômalo durante procedimentos.
- Objetos suspeitos que possam indicar automutilação.

Relatórios automáticos especializados:

- Desvios em procedimentos obstétricos.
- Sinais de complicações em cirurgias ginecológicas.
- Indicadores visuais de desconforto psicológico.
- Alertas para possíveis casos de violência doméstica.

#### 2. Análise de Áudio Especializada para Saúde da Mulher

Processar áudios de consultas médicas especializadas:

- Consultas ginecológicas: tom de voz, hesitação ao relatar sintomas.
- Acompanhamento pré-natal: sinais de ansiedade gestacional.
- Consultas pós-parto: detecção precoce de depressão pós-parto.
- Atendimento a vítimas de violência: padrões vocais indicativos de trauma.

## Entregáveis da Fase 4

### Repositório Git

- Código-fonte completo da solução.
- Relatório técnico com:
  - Descrição do fluxo multimodal.
  - Modelos aplicados em cada tipo de dado.
  - Resultados obtidos e exemplos de anomalias detectadas.

### Vídeo com até 15 minutos demonstrando

- Upload no YouTube ou Vimeo (público ou não listado).
- Demonstração do processamento multimodal:
  - Exemplo prático da análise de áudio e vídeo.
  - Detecção e resposta a anomalias.
  - Integração dos serviços Azure.
  - Fluxo final do alerta à equipe médica.
