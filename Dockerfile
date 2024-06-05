FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install -U pip setuptools wheel

RUN pip install -r requirements.txt

CMD ["python", "app.py"]
