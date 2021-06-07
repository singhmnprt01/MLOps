FROM python:3.8-slim-buster
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
COPY ./rf_survival.pkl /app/rf_survival.pkl
COPY ./model_survival_columns.pkl /app/model_survival_columns.pkl
RUN pip install -r requirements.txt
COPY ./app.py /app
ENTRYPOINT ["python"]
CMD ["app.py"]