FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# (Opcional) para gráficos com matplotlib salvando em arquivos:
ENV MPLBACKEND=Agg

COPY . .
# Executa o pipeline principal
CMD ["python", "src/main.py"]
