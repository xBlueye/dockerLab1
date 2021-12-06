FROM python:3.7
WORKDIR /app

COPY . .

RUN pip3 install --upgrade pip
RUN pip install -r requirements.txt


ENTRYPOINT ["python","Project.py"]