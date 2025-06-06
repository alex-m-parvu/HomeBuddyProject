import pandas as pd
from DS_Helpers.filters import column_filter
from DS_Helpers.data import other_generator_columns, find_embed_object_cols, get_days_to_nearest_holiday
from DS_Helpers.metrics import classification_result
from DS_Helpers.models import simple_preprocessing, log_scaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Define machine learning pipeline
pipe = make_pipeline(
    simple_preprocessing,               # Custom preprocessing step
    StandardScaler(),                   # Feature standardization
    log_scaler,                         # Custom log transformation for outliers squashing
    RandomForestClassifier(
        n_estimators=200,               # Number of trees
        n_jobs=-1                       # Use all cores
    )
)

class pred_pipe:
    """
    A predictive modeling pipeline for lead appointment setting.
    
    Handles loading, cleaning, merging, feature engineering, training, and evaluation.
    """

    def __init__(self, 
                 leads='data/leads_dataset.csv',
                 infutor_enrichment_dataset='data/infutor_enrichment_dataset.csv',
                 zip_code_dataset='data/zip_code_dataset.csv',
                 country_code='US'):
        """
        Initializes the pipeline by loading and merging datasets, 
        cleaning columns, and generating holiday distance feature.
        
        Args:
            leads (str): Path to leads dataset.
            infutor_enrichment_dataset (str): Path to enrichment dataset.
            zip_code_dataset (str): Path to zip code dataset.
            country_code (str): Country code for holiday distance calculation.
        """
        # Load and filter leads dataset
        leads = column_filter(pd.read_csv(leads), tolerance=0.4)
        leads.LEAD_CREATED_AT_UTC = pd.to_datetime(leads.LEAD_CREATED_AT_UTC)
        self.leads = leads

        # Load and filter enrichment and zip datasets
        self.infutor_enrichment_dataset = column_filter(pd.read_csv(infutor_enrichment_dataset, index_col=0), tolerance=0.4)
        self.zip_code_dataset = column_filter(pd.read_csv(zip_code_dataset), tolerance=0.3)

        # Merge datasets on key columns
        self.data = pd.merge(self.leads, self.infutor_enrichment_dataset, on='HASHED_PHONE_NUMBER', how='left')
        self.data = pd.merge(self.data, self.zip_code_dataset, on='ZIP_CODE', how='left')

        # Add days to nearest holiday
        print('Generating Holiday Distance. Please wait this could take up to a few minutes')
        self.data['days_to_holiday'] = self.data['LEAD_CREATED_AT_UTC'].apply(
            lambda x: get_days_to_nearest_holiday(x, country_code=country_code) if pd.notnull(x) else None
        )
        print('Holiday Distance Created!')

        # Drop rows without target value and remove ID columns
        self.data = self.data[~self.data.IS_APPOINTMENT_SET.isna()]
        self.data = self.data.drop(labels=['HASHED_PHONE_NUMBER', 'ZIP_CODE'], axis=1)

    def train_test_split(self, test_size=0.3, random_state=42, embedding_dims=8):
        """
        Splits the data into training and test sets, extracts date features, 
        applies "other" category consolidation, and performs embedding on categorical features.
        
        Args:
            test_size (float): Proportion of test set.
            random_state (int): Seed for reproducibility.
            embedding_dims (int): Embedding dimension for categorical variables.
        """
        # Define target and features
        y = self.data.IS_APPOINTMENT_SET
        X = self.data.drop(labels='IS_APPOINTMENT_SET', axis=1)

        # Extract date/time components
        X['Year'] = X.LEAD_CREATED_AT_UTC.dt.year
        X['Month'] = X.LEAD_CREATED_AT_UTC.dt.month.astype(str)
        X['Weekday'] = X.LEAD_CREATED_AT_UTC.dt.weekday.astype(str)
        X['Day'] = X.LEAD_CREATED_AT_UTC.dt.day.astype(str)
        X['Hour'] = X.LEAD_CREATED_AT_UTC.dt.hour.astype(str)
        X = X.drop(labels='LEAD_CREATED_AT_UTC', axis=1)

        self.X, self.y = X, y

        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        # Replace infrequent values in categorical features
        self.X_train = other_generator_columns(df=self.X_train, other_threshold=50)
        self.X_test = other_generator_columns(df=self.X_test, other_threshold=50)

        # Learn and apply embeddings on training set
        self.obj_embed = find_embed_object_cols(df=self.X_train, embedding_dims=embedding_dims)
        self.X_train_embed = self.obj_embed['df']

        # Apply embeddings to test set
        self.X_test_embeding = self.X_test.copy()
        embedings = self.obj_embed['embedings']
        for key in embedings.keys():
            emb_df = embedings[key].transform(self.X_test_embeding[key])
            emb_df.index = self.X_test_embeding.index
            self.X_test_embeding = pd.concat([self.X_test_embeding, emb_df], axis=1)
            self.X_test_embeding = self.X_test_embeding.drop(labels=key, axis=1)

    def train(self, pipe=pipe):
        """
        Trains the model using the defined pipeline on the training data.

        Args:
            pipe (Pipeline): Predefined sklearn pipeline.
        """
        self.pipe = pipe
        self.pipe.fit(self.X_train_embed, self.y_train.astype(str))

    def evaluate_model(self):
        """
        Evaluates the model using precision-recall curve on the test set.
        """
        probs = self.pipe.predict_proba(self.X_test_embeding)
        precision, recall, _ = precision_recall_curve(self.y_test.astype(int), probs[:, 1])
        plt.plot(precision, recall)
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.show()

    def evaluate_model_metrics(self):
        """
        Evaluates and prints detailed classification metrics on the test set.
        """
        self.y_pred = self.pipe.predict(self.X_test_embeding)
        classification_result(y_true=self.y_test.astype(str), y_pred=self.y_pred)
