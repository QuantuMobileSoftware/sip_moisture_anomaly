FROM python:3.8.15-slim

# Needed for geopandas&shapely to work
RUN apt-get update && \
    apt-get install -y \
    git \
    libspatialindex-dev \
    libgl1-mesa-glx

RUN pip install jupyter==1.0.0

COPY ./requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt

WORKDIR /code
COPY ./ .

CMD ["jupyter", "nbconvert", "--inplace", "--to=notebook", "--execute", "./moisture_anomaly_calculation.ipynb"]
