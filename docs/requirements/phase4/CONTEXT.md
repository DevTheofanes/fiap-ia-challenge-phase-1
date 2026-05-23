# CONTEXT — Fase 4

Glossário canônico de termos do domínio para a Fase 4 do Tech Challenge. Este arquivo é uma referência de linguagem — não contém decisões de implementação.

---

## Termos

### Análise Multimodal
Processamento combinado de dados de áudio e vídeo clínico para apoio à decisão médica. Distinto da análise de texto já presente na Fase 3.

### Sangramento Anômalo
Presença de sangue em quantidade ou localização atípica durante procedimentos cirúrgicos ginecológicos ou obstétricos. Alvo de detecção do modelo YOLOv8 customizado.

### Frame Sintético
Imagem gerada programaticamente para treino do YOLOv8. Simula frames de laparoscopia com manchas vermelhas sobrepostas representando sangramento, com anotações YOLO geradas automaticamente no mesmo processo.

### Pipeline de Áudio
Fluxo composto por duas etapas sequenciais: (1) transcrição da fala via Whisper API e (2) análise clínica do transcript via GPT-4o-mini. Produz um laudo textual estruturado.

### Transcript
Saída textual da transcrição de um áudio de consulta médica via Whisper API. Entrada para a análise clínica do GPT-4o-mini.

### Laudo de Áudio
Relatório clínico estruturado gerado pelo GPT-4o-mini a partir do transcript. Contém indicadores de: depressão pós-parto, ansiedade gestacional, sinais de violência doméstica e fadiga hormonal.

### Laudo de Vídeo
Relatório gerado pelo pipeline YOLOv8 após inferência em um vídeo clínico. Contém: frames com detecções, bounding boxes, confidence scores e classificação de risco.

### Estado Multimodal
Extensão do `AssistantState` do LangGraph com os campos: `audio_path`, `audio_transcript`, `audio_analysis`, `video_path`, `video_report`. Campos opcionais — nós pulam processamento se não fornecidos.

### Fine-tune
Processo de ajuste dos pesos do modelo `yolov8n.pt` (pré-treinado no COCO) usando frames sintéticos de sangramento. Produz um modelo especializado na classe `anomalous_bleeding`.

### OpenAIClient
Novo cliente LLM adicionado ao projeto, seguindo o padrão de `GeminiClient`. Usado exclusivamente para a análise clínica do áudio via GPT-4o-mini e para a transcrição via Whisper API.
