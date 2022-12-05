FROM python:3.8.15-slim-buster

ADD ./ /inductive-knowledge-service/

WORKDIR /inductive-knowledge-service

RUN pip install -r requirements.txt

ENTRYPOINT [ "python" ]

CMD ["NLPController.py" ]