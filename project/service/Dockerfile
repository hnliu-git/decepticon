FROM python:3

COPY requirements.txt ./

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade --no-cache-dir -r requirements.txt

ARG APP_ROOT=/service

COPY ./ckpts ${APP_ROOT}/ckpts
COPY ./*.py ${APP_ROOT}/
COPY ./templates ${APP_ROOT}/templates

ENV APP_ROOT ${APP_ROOT}

ARG PORT=8333
ENV PORT ${PORT}
EXPOSE ${PORT}

ENV FLASK_APP /service/serve

CMD flask run --host 0.0.0.0 --port ${PORT}

