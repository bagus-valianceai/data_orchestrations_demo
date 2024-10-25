from datetime import datetime
from airflow.decorators import task
from credit_scoring_service.utils import utils

@task(task_id = "extract_credit_data")
def extract_credit_data(**kwargs):
    # Get Task Instance for XCOM communication
    ti = kwargs["ti"]

    # Logging to indicate task being executed
    print(f"Executing task_id: {ti.task_id}.")

    # Get the last date of extracted credit data from database. If there is new data in database, then extract it to retrain model
    last_extracted_date = utils.variable_do(
        method = "get",
        key = "last_extracted_credit_data"
    )
    print(f"Last extracted credit data: {last_extracted_date}")

    # Connect to database
    _, cursor = utils.connect_database(db_conn_id = "credit-data-db-conn")

    # Condition if there is no data in airflow variable for 'last_extracted_date', this is important to query the whole dataset for training
    if(last_extracted_date == None):
        print(f"Getting first date of credit data in database.")
        cursor.execute("""
                   SELECT created_at
                   FROM data_credit
                   ORDER BY created_at ASC
                   LIMIT 1;
                   """)
        last_extracted_date = cursor.fetchall()[0][0]
        print(f"First date of credit data in database is {last_extracted_date}")

        # Store the data in airflow variable
        utils.variable_do(
            method = "set",
            key = "last_extracted_credit_data",
            data = last_extracted_date
        )

    # Get latest date of credit data in database
    cursor.execute("""
                   SELECT created_at
                   FROM data_credit
                   ORDER BY created_at DESC
                   LIMIT 1;
                   """)
    latest_credit_data_date = cursor.fetchall()[0][0]
    print(f"Lastest credit data: {latest_credit_data_date}")

    # Convert from string date to datetime data type
    last_extracted_date = datetime.strptime(last_extracted_date, "%Y-%m-%d")
    latest_credit_data_date = datetime.strptime(latest_credit_data_date, "%Y-%m-%d")

    # Condition if last extracted date is less than lastest date in database, meaning there are new data
    if(last_extracted_date < latest_credit_data_date):
        print(f"Newer data available, delta time is: {latest_credit_data_date - last_extracted_date}")

        # Get data from last extracted to the latest data
        cursor.execute(f"""
                    SELECT *
                    FROM data_credit
                    WHERE created_at
                    BETWEEN '{last_extracted_date.strftime("%Y-%m-%d")}'
                    AND '{latest_credit_data_date.strftime("%Y-%m-%d")}'
                    ORDER BY created_at ASC;
                    """)
        newest_credit_data = cursor.fetchall()

        # Store current date of extracted data to airflow variable so next time it runs it will not query from beginning
        utils.variable_do(
            method = "set", 
            key = "last_extracted_credit_data",
            data = newest_credit_data[-1][-1]
        )

        # Pushing query result to MinIO in pickle format (right now only supported pickle format)
        utils.minio_do(
            method = "push",
            key = f"extraction_{newest_credit_data[-1][-1].replace("-", "")}.pkl",
            bucket_name = "credit-scoring-service",
            data = newest_credit_data
        )

        # Push the filename of query result to XCOM so next task could used it
        # Large data isn't recommended to be pushed to XCOM 
        utils.xcom_do(
            ti = ti,
            method = "push",
            key = "extracted_data_filename",
            data = f"extraction_{newest_credit_data[-1][-1].replace("-", "")}.pkl"
        )

        # Pushing query result to MinIO in pickle format (right now only supported pickle format)
        utils.minio_do(
            method = "push",
            key = f"extraction_{newest_credit_data[-1][-1].replace("-", "")}_colnames.pkl",
            bucket_name = "credit-scoring-service",
            data = [desc[0] for desc in cursor.description]
        )

        # Push the filename of query result to XCOM so next task could used it
        # Large data isn't recommended to be pushed to XCOM 
        utils.xcom_do(
            ti = ti,
            method = "push",
            key = "extracted_data_colnames",
            data = f"extraction_{newest_credit_data[-1][-1].replace("-", "")}_colnames.pkl"
        )
    
    # Condition when there is no new dat available in database
    else:
        print(f"No new data available, delta time is: {latest_credit_data_date - last_extracted_date}")