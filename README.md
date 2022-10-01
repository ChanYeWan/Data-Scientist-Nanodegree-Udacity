# Disaster Response Pipeline Project

Project Overview

Build a model for an API that classifies real disaster messages from Figure Eight so that the messages will be delivered to an appropriate disaster agency.

### Instructions:
1. Run the following commands in the project's root directory to set up database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the commands of `cd app` and `python run.py`

4. Open http://0.0.0.0:3000/
