FROM python:3.12.0
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

RUN python procesing.py
CMD [ "python", "main.py" ]
