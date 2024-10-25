from airflow.decorators import task
from credit_scoring_service.utils import utils
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

@task(task_id = "training_credit_data")
def training_credit_data(**kwargs):
    ti = kwargs["ti"]

    print("Pulling dataset, encoder, imputer, and scaler from MinIO.")

    # Pull train set
    trainset = utils.xcom_do(
        ti = ti,
        method = "pull",
        key = "preprocessed_trainset_filename",
        task_ids = "preprocess_credit_data",
        include_prior_dates = True
    )
    trainset = utils.minio_do(
        method = "pull",
        key = trainset,
        bucket_name = "credit-scoring-service"
    )
    
    # Pull valid set
    validset = utils.xcom_do(
        ti = ti,
        method = "pull",
        key = "preprocessed_validset_filename",
        task_ids = "preprocess_credit_data",
        include_prior_dates = True
    )
    validset = utils.minio_do(
        method = "pull",
        key = validset,
        bucket_name = "credit-scoring-service"
    )

    # Pull test set
    testset = utils.xcom_do(
        ti = ti,
        method = "pull",
        key = "preprocessed_testset_filename",
        task_ids = "preprocess_credit_data",
        include_prior_dates = True
    )
    testset = utils.minio_do(
        method = "pull",
        key = testset,
        bucket_name = "credit-scoring-service"
    )

    # Pull numerical imputer
    num_imputer = utils.xcom_do(
        ti = ti,
        method = "pull",
        key = "preprocessed_num_imputer_filename",
        task_ids = "preprocess_credit_data",
        include_prior_dates = True
    )
    num_imputer = utils.minio_do(
        method = "pull",
        key = num_imputer,
        bucket_name = "credit-scoring-service"
    )

    # Pull categorical imputer
    cat_imputer = utils.xcom_do(
        ti = ti,
        method = "pull",
        key = "preprocessed_cat_imputer_filename",
        task_ids = "preprocess_credit_data",
        include_prior_dates = True
    )
    cat_imputer = utils.minio_do(
        method = "pull",
        key = cat_imputer,
        bucket_name = "credit-scoring-service"
    )

    # Pull one hot encoder
    ohe_encoder = utils.xcom_do(
        ti = ti,
        method = "pull",
        key = "preprocessed_ohe_filename",
        task_ids = "preprocess_credit_data",
        include_prior_dates = True
    )
    ohe_encoder = utils.minio_do(
        method = "pull",
        key = ohe_encoder,
        bucket_name = "credit-scoring-service"
    )
    
    # Pull scaler
    scaler = utils.xcom_do(
        ti = ti,
        method = "pull",
        key = "preprocessed_scaler_filename",
        task_ids = "preprocess_credit_data",
        include_prior_dates = True
    )
    scaler = utils.minio_do(
        method = "pull",
        key = scaler,
        bucket_name = "credit-scoring-service"
    )

    print("Start training model.")

    # Training model
    model = DecisionTreeClassifier()
    model.fit(trainset[0], trainset[1])

    print("Training model completed.")

    # Get the date of data extracted
    last_extracted_credit_data = utils.variable_do(method = "get", key = "last_extracted_credit_data").replace("-", "")

    print("Pushing to MinIO for model versioning.")

    # Push trained model to MinIO for versioning
    utils.minio_do(
        method = "push",
        key = f"training_model_{last_extracted_credit_data}.pkl",
        bucket_name = "credit-scoring-service",
        data = model
    )
    utils.xcom_do(
        ti = ti,
        method = "push",
        key = "trained_model_filename",
        data = f"training_model_{last_extracted_credit_data}.pkl"
    )

    print("New trained model has been pushed.")

    # Check if there is previous best model
    prev_bestmodel_filename = utils.variable_do(
        method = "get",
        key = "prev_best_model"
    )

    # Push again current model as best model if there is no previous best model
    if(prev_bestmodel_filename == None):
        print("No previous best model, current model marked as best model.")

        utils.minio_do(
            method = "push",
            key = f"best_model.pkl",
            bucket_name = "credit-scoring-service",
            data = model
        )
        utils.variable_do(
            method = "set",
            key = "prev_best_model",
            data = "best_model.pkl"
        )
    
    # Condition if there is previously best model
    else:
        print("Previous best model detected.")

        prev_best_model = utils.minio_do(
            method = "pull",
            key = "best_model.pkl",
            bucket_name = "credit-scoring-service"
        )

        y_pred = prev_best_model.predict(testset[0])
        prev_best_model_f1 = classification_report(testset[1], y_pred, output_dict = True)["macro avg"]["f1-score"]

        y_pred = model.predict(testset[0])
        current_model_f1 = classification_report(testset[1], y_pred, output_dict = True)["macro avg"]["f1-score"]

        print("Comparing previous and current model's performance.")
        print("Previous model performance (F1-Score): ", prev_best_model_f1)
        print("Current model metrics (F1-Score): ", current_model_f1)

        if(prev_best_model_f1 < current_model_f1):
            print("Current model is the new best model.")

            utils.minio_do(
                method = "push",
                key = f"best_model.pkl",
                bucket_name = "credit-scoring-service",
                data = model
            )
            utils.variable_do(
                method = "set",
                key = "prev_best_model",
                data = "best_model.pkl"
            )

        else:
            print("Previous model still the best model.")