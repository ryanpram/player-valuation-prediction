FROM python:3.9-slim

# Install all the dependencies from the Pipenv file
RUN pip install pipenv

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

# Copy your Flask script
COPY ["predict.py", "model_v1.bin", "./"]

# Run it with Gunicorn
EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]