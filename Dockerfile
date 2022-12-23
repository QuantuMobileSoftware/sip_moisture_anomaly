FROM python:3.8.15-slim

# Needed for geopandas&shapely to work
RUN apt-get update && \
    apt-get install -y \
    git \
    libspatialindex-dev \
    libgl1-mesa-glx

RUN mkdir /code

WORKDIR /code
COPY ./ .

RUN pip install jupyter==1.0.0
RUN pip install -r requirements.txt

CMD ["jupyter", "nbconvert", "--inplace", "--to=notebook", "--execute", "./water_anomaly_calculation.ipynb"]
