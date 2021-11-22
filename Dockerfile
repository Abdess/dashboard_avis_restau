FROM python:3.9
LABEL maintainer="Abdessamad DERRAZ <derraz.abdessamad@gmail.com>"

WORKDIR /usr/src/app
COPY .. .
RUN pip install -U pip && pip install --no-cache-dir -e .
EXPOSE 8501
ENTRYPOINT streamlit run ./dashboard_avis_restau/app.py