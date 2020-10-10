FROM python:3.7-stretch@sha256:ba2b519dbdacc440dd66a797d3dfcfda6b107124fa946119d45b93fc8f8a8d77

WORKDIR /app

RUN apt-get clean \
    && apt-get -y update

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir pytest

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV LANG C.UTF-8

CMD [ "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]