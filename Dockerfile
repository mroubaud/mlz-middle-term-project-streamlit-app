#instalo imagen base de python
FROM python

#instalo pipenv para poder correr los archivos Pipfile
RUN pip install pipenv

#dentro de la imagen docker, creo y me paro 
#en el directorio /app
WORKDIR /app

#copio los archivos Pipfile en el directorio actual
# es decir, en /app
COPY ["Pipfile", "Pipfile.lock", "./"]

#instalo dependencias de los archivos Pipfile pero 
#sin crear virtual enviorment, la imagen docker ya me aisla
RUN pipenv install --system --deploy

#Copio los archivos para predecir y el de final_model en
#el directorio actual (/app) 
COPY ["predict.py", "final_model.bin", "dv.bin", "./"]

#expongo puerto 9696 de la imagen para recibir requests
EXPOSE 9696

#ejecuto comando para correr servidor gunicorn en
#localhost y  escuchando puerto 9696
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]

