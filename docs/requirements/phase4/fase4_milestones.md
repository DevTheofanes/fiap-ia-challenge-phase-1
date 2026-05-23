# Milestones — Fase 4

## Decisões de arquitetura

- Modalidades: Vídeo + Áudio
- Alvo YOLOv8: detecção de sangramento anômalo (`anomalous_bleeding`)
- Dados YOLOv8: frames sintéticos gerados programaticamente
- Estratégia YOLOv8: fine-tune sobre `yolov8n.pt` (~20 épocas)
- Transcrição: Whisper API (OpenAI)
- Análise clínica: GPT-4o-mini via `OpenAIClient`
- Integração: novos nós `process_audio` e `process_video` no LangGraph existente
- Formato: módulos em `src/multimodal/`, scripts em `scripts/`, demo em `notebooks/`

---

## M1 — Novo cliente LLM (OpenAI)

- [x] Adicionar `OpenAIClient` em `src/llm/client.py` seguindo o padrão de `GeminiClient`
- [x] Adicionar `OPENAI_API_KEY` e `OPENAI_MODEL` ao `.env.example`
- [x] Estender `get_llm_client()` para retornar `OpenAIClient` quando `OPENAI_API_KEY` estiver definido

## M2 — Pipeline de áudio

- [x] Criar `src/multimodal/__init__.py`
- [x] Criar `src/multimodal/audio.py`
  - [x] Função `transcribe(audio_path)` — chama Whisper API, retorna transcript
  - [x] Função `analyze_transcript(transcript, client)` — envia ao GPT-4o-mini, retorna laudo clínico estruturado
  - [x] Prompt clínico cobrindo: depressão pós-parto, ansiedade, sinais de violência, fadiga hormonal
- [x] Criar `scripts/analyze_audio.py` — CLI standalone para testar o pipeline

## M3 — Geração de dados sintéticos para YOLOv8

- [x] Criar `scripts/generate_synthetic_data.py`
  - [x] Gerar N frames base (fundo escuro simulando cavidade laparoscópica)
  - [x] Sobrepor manchas vermelhas em posições aleatórias (simulação de sangramento)
  - [x] Gerar labels YOLO (`.txt`) automaticamente com as coordenadas das manchas
  - [x] Gerar `dataset.yaml` com classes: `['anomalous_bleeding']`
  - [x] Salvar em `data/synthetic_bleeding/`

## M4 — Fine-tune YOLOv8

- [x] Adicionar `ultralytics` ao `requirements.txt`
- [x] Criar `scripts/train_yolo.py`
  - [x] Fine-tune `yolov8n.pt` com os dados sintéticos (~20 épocas)
  - [x] Salvar modelo treinado em `artifacts/yolo/`
  - [x] Logar métricas: mAP50, precision, recall

## M5 — Pipeline de vídeo

- [x] Criar `src/multimodal/video.py`
  - [x] Função `analyze_video(video_path, model_path)` — roda YOLOv8 frame a frame
  - [x] Função `generate_video_report(detections)` — retorna laudo textual estruturado com contagem de detecções, confidence médio e classificação de risco
- [x] Criar `scripts/analyze_video.py` — CLI standalone para testar o pipeline

## M6 — Integração com LangGraph

- [x] Estender `AssistantState` em `src/assistant/graph.py` com campos:
  - `audio_path: str | None`
  - `audio_transcript: str | None`
  - `audio_analysis: str | None`
  - `video_path: str | None`
  - `video_report: str | None`
- [x] Adicionar nó `process_audio` — pula se `audio_path` for None
- [x] Adicionar nó `process_video` — pula se `video_path` for None
- [x] Inserir os novos nós entre `retrieve_patient_context` e `generate_response`
- [x] Estender `build_generation_prompt` em `src/assistant/chain.py` para incluir `audio_analysis` e `video_report` no contexto
- [x] Estender `scripts/run_assistant.py` com flags `--audio` e `--video`

## M7 — Notebook de demonstração

- [ ] Criar `notebooks/demo_phase4.ipynb` cobrindo:
  - [ ] Demonstração do pipeline de áudio (áudio sintético → transcript → laudo)
  - [ ] Demonstração do pipeline de vídeo (vídeo sintético → detecções → relatório)
  - [ ] Fluxo integrado via LangGraph com áudio + vídeo + pergunta clínica

## M8 — Relatório técnico e entregáveis

- [ ] Criar `docs/reports/Relatorio_Tecnico_Tech_Challenge_Fase4.md` com:
  - [ ] Descrição do fluxo multimodal
  - [ ] Modelos aplicados por modalidade
  - [ ] Resultados e exemplos de anomalias detectadas
  - [ ] Métricas do YOLOv8 (mAP, precision, recall)
- [ ] Atualizar `README.md` com instruções da Fase 4
- [ ] Gravar vídeo de demonstração (máx. 15 min) e adicionar link no README
