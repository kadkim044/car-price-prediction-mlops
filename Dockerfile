FROM python:3.14.0

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 9696

CMD ["uvicorn","src.api:app","--host","0.0.0.0","--port","9696"]