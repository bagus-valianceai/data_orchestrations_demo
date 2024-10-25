import pandas as pd
from airflow.decorators import task
from credit_scoring_service.utils import utils, preprocess_util

@task(task_id = "preprocess_credit_data")
def preprocess_credit_data(**kwargs):
    ti = kwargs["ti"]

    print("Pulling data and data's column names from MinIO.")

    # Get the file name of data previously pushed to MinIO so this task could pull the data from MinIO
    extracted_data_filename = utils.xcom_do(
        ti = ti,
        method = "pull",
        task_ids = "extract_credit_data",
        key = "extracted_data_filename",
        include_prior_dates = True
    )

    # Also get the file containing list of column names
    extracted_data_colnames = utils.xcom_do(
        ti = ti,
        method = "pull",
        task_ids = "extract_credit_data",
        key = "extracted_data_colnames",
        include_prior_dates = True
    )

    # Pulling the new data from MinIO
    new_extracted_data = utils.minio_do(
        method = "pull",
        key = extracted_data_filename,
        bucket_name = "credit-scoring-service"
    )
    
    # Pulling the list of column name from MinIO
    new_extracted_data_colnames = utils.minio_do(
        method = "pull",
        key = extracted_data_colnames,
        bucket_name = "credit-scoring-service"
    )

    print("Start preprocessing data.")

    # Create DataFrame of those 2 parts in order to be preprocessed further
    dataset = pd.DataFrame(new_extracted_data, columns = new_extracted_data_colnames)
    
    # Split columnwise dataset into features (input) and target (output)
    X, y = preprocess_util.split_input_output(data = dataset, target_col = "loan_status")

    # Split rowwise dataset into train, valid, and test set
    X_train, X_not_train, y_train, y_not_train = preprocess_util.split_train_test(
        X = X,
        y = y,
        test_size = 0.2,
        random_state = 42
    )
    X_valid, X_test, y_valid, y_test = preprocess_util.split_train_test(
        X = X_not_train,
        y = y_not_train,
        test_size = 0.5,
        random_state = 42
    )

    # Fit inputer, encoder, andscaler
    num_imputer, cat_imputer, ohe_encoder, scaler = preprocess_util.fit_preprocess_data(X_train)

    # Retransform the train set data
    X_train_clean = preprocess_util.transform_preprocess_data(
        X = X_train,
        num_imputer = num_imputer,
        cat_imputer = cat_imputer,
        ohe_encoder = ohe_encoder,
        scaler = scaler
    )

    # Transform the valid set data
    X_valid_clean = preprocess_util.transform_preprocess_data(
        X = X_valid,
        num_imputer = num_imputer,
        cat_imputer = cat_imputer,
        ohe_encoder = ohe_encoder,
        scaler = scaler
    )

    # Transform the test set data
    X_test_clean = preprocess_util.transform_preprocess_data(
        X = X_test,
        num_imputer = num_imputer,
        cat_imputer = cat_imputer,
        ohe_encoder = ohe_encoder,
        scaler = scaler
    )

    print("Preprocessing data completed.")

    # Get the date of data extracted
    last_extracted_credit_data = utils.variable_do(method = "get", key = "last_extracted_credit_data").replace("-", "")

    print("Pushing to MinIO.")

    # Push train set
    utils.minio_do(
        method = "push",
        key = f"preprocess_trainset_{last_extracted_credit_data}.pkl",
        bucket_name = "credit-scoring-service",
        data = [X_train_clean, y_train]
    )
    utils.xcom_do(
        ti = ti,
        method = "push",
        key = "preprocessed_trainset_filename",
        data = f"preprocess_trainset_{last_extracted_credit_data}.pkl"
    )

    # Push valid set
    utils.minio_do(
        method = "push",
        key = f"preprocess_validset_{last_extracted_credit_data}.pkl",
        bucket_name = "credit-scoring-service",
        data = [X_valid_clean, y_valid]
    )
    utils.xcom_do(
        ti = ti,
        method = "push",
        key = "preprocessed_validset_filename",
        data = f"preprocess_validset_{last_extracted_credit_data}.pkl"
    )

    # Push test set
    utils.minio_do(
        method = "push",
        key = f"preprocess_testset_{last_extracted_credit_data}.pkl",
        bucket_name = "credit-scoring-service",
        data = [X_test_clean, y_test]
    )
    utils.xcom_do(
        ti = ti,
        method = "push",
        key = "preprocessed_testset_filename",
        data = f"preprocess_testset_{last_extracted_credit_data}.pkl"
    )

    # Push numerical imputer
    utils.minio_do(
        method = "push",
        key = f"preprocess_num_imputer_{last_extracted_credit_data}.pkl",
        bucket_name = "credit-scoring-service",
        data = num_imputer
    )
    utils.xcom_do(
        ti = ti,
        method = "push",
        key = "preprocessed_num_imputer_filename",
        data = f"preprocess_num_imputer_{last_extracted_credit_data}.pkl"
    )

    # Push categorical imputer
    utils.minio_do(
        method = "push",
        key = f"preprocess_cat_imputer_{last_extracted_credit_data}.pkl",
        bucket_name = "credit-scoring-service",
        data = cat_imputer
    )
    utils.xcom_do(
        ti = ti,
        method = "push",
        key = "preprocessed_cat_imputer_filename",
        data = f"preprocess_cat_imputer_{last_extracted_credit_data}.pkl"
    )

    # Push one hot encoder
    utils.minio_do(
        method = "push",
        key = f"preprocess_ohe_{last_extracted_credit_data}.pkl",
        bucket_name = "credit-scoring-service",
        data = ohe_encoder
    )
    utils.xcom_do(
        ti = ti,
        method = "push",
        key = "preprocessed_ohe_filename",
        data = f"preprocess_ohe_{last_extracted_credit_data}.pkl"
    )
    
    # Push scaler
    utils.minio_do(
        method = "push",
        key = f"preprocess_scaler_{last_extracted_credit_data}.pkl",
        bucket_name = "credit-scoring-service",
        data = scaler
    )
    utils.xcom_do(
        ti = ti,
        method = "push",
        key = "preprocessed_scaler_filename",
        data = f"preprocess_scaler_{last_extracted_credit_data}.pkl"
    )

    print("Trainset, validset, testset, numerical imputer, categorical imputer, ohe, and scaler has been pushed to MinIO.")