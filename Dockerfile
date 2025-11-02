FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# (Opcional) para gráficos com matplotlib salvando em arquivos:
ENV MPLBACKEND=Agg

COPY . .
# Exemplo: rodar um script que treina e salva resultados
CMD ["python", "src/models.py"]
