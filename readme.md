# Repository for Course Data Orchestrations
## Prerequisite
The Apache Airflow it self could be started from docker-compose.yaml in the `docker` directory, but for the course you must setup **PostgreSQL** and **MinIO**.
<br><br>
The `docker-compose.yaml` for postgres and minio are already provided in directory `postgres` and `minio`.
<br><br>
## How to Setup PostgreSQL
1. Change directory to `postgres`
2. Open the `.env` file
3. Change value of key `POSTGRES_USER` to setup your postgres username
4. Change value of key `POSTGRES_PASS` to setup your postgres password
5. Save the `.env` file
6. Open terminal and execute command `docker compose up -d` to start postgres
7. Postgres should be accepting connection in `locahost` and port `5432`
<br><br>
## How to Setup MinIO
1. Change directory to `minio`
2. Open the `.env` file
3. Change value of key `MINIO_ROOT_USER` to setup your minio username
4. Change value of key `MINIO_ROOT_PASSWORD` to setup your minio password
5. Save the `.env` file
6. Open terminal and execute command `docker compose up -d` to start minio
7. Minio should be accepting connection in `localhost` and port `9001`
8. Don't forget to create bucket, this course would use bucket name `credit-scoring-service` to interchange data between Airflow Task
<br><br>
## How to Setup Airflow
1. Create empty directory named `config`, `logs`, `dags`, and `plugins` if they not existed
2. In directory `docker`, edit the `.env` file and change the value of key `_AIRFLOW_WWW_USER_USERNAME` in order to setup your username
3. In directory `docker`, edit the `.env` file and change the value of key `_AIRFLOW_WWW_USER_PASSWORD` in order to setup your password
4. In directory `docker`, edit the `.env` file and change the value of key `_PIP_ADDITIONAL_REQUIREMENTS` if you have package that would be needed for your application
5. Open terminal, change directory to `docker` where the `docker-compose.yaml` is located and execute command `docker compose up -d`
6. Wait for several minutes and then open `localhost:8081` to access Apache Airflow UI
<br><br>
## How to Run Model's API Endpoints
This API Endpoints intended for testing prediction of model that has been trained, if you not trained your model yet you can't start this API Endpoints because the required files isn't available.
<br><br>
The required files:
1. `best_model.pkl`
2. `preprocess_cat_imputer_[yyyymmdd].pkl`
3. `preprocess_num_imputer_[yyyymmdd].pkl`
4. `preprocess_ohe_[yyyymmdd].pkl`
5. `preprocess_scaler_[yyymmdd].pkl`
<br><br>
The **best_model.pkl** is trained model.
<br>
The **preprocess_cat_imputer_[yyyymmdd].pkl** and **preprocess_num_imputer_[yyyymmdd].pkl** are categorical and numerical imputer, in the last part of name **[yyyymmdd]** is the date when imputers are fitted. Using different date of imputer with model could broke your pipeline.
<br>
The **preprocess_ohe_[yyyymmdd].pkl** is encoder for categorical data, the **[yyyymmdd]** part is the same as imputer.
<br>
The **preprocess_scaler_[yyymmdd].pkl** is the feature scaler for data numerical.
<br><br>
All required files could be downloaded manually from MinIO WebUI after you train the model.
<br><br>
This API Endpoints also required Python Virtual Environment.
How to setup VENV:
1. Go to root directory
2. Open terminal and execute command `python3 -m venv .venv_credit_scoring`
3. For WSL, Ubuntu, and Linux, activate the venv by executing command `source .venv_credit_scoring/bin/activate`
4. Install requirements by executing command `pip install -r requirements.txt`, make sure you are in the same folder with the `requirements.txt` file
<br><br>
How to run:
1. After all files above has been provided and venv has been acticated, change directory to `api/src`
2. Open terminal and execute command `fastapi dev api.py
3. Open web browser and go to `localhost:8000`
4. Open `localhost:8000/docs` if you intended to test the API Endpoints