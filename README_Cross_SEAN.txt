Cross-SEAN



Folder Contents

data.zip - Contains sample data (2000 Fake, 2000 Genuine and 5000 Unlabelled) according to the format expected by the pipeline.

covid19.sh - Shell config file which ensures the end-to-end working.

extract_features.py - Prepares the data by extracting various inputs from the tweet objects containing json files.

preprocess.py - Prepares all the data according to the embedding vector inputs.

model.py - Contains all the model component classes

training.py - Runs the training epochs and evaluation steps

main.py - Loads all the data and initialises training, followed by evaluation

config.py - Contains all args which are parsed after running main.py, along with default values.

flask_sean.py - Responsible for the chrome extension API (to run, replace your Tweepy credentials in the file)

sentimentanalysis - A helper module to generate some tweet features in ```preprocess.py```. Uses a slightly modified version of [williamscott701/sentimentanalysis](https://github.com/williamscott701/sentimentanalysis) to create the extended version of the data.


Compute Requirements:
- 5GB GPU for training
- 1GB for inference


How to run?
- Unzip data.zip
OR
- Add/Update data in data/covid19/ in json format, as fake.json, genuine.json, unlabelled.json. (sample data given in CTF are given here in json format)
- Install the requirements
        ```
        pip install -r requirements.txt
        ```
- Change the project and directory name in covid19.sh (optional)
- Run the pipeline (changing the paths appropriately, if any of the above steps are done differently)
        ```
        sh ./covid19.sh
        ```

Note
- To run live service, follow README_Chrome_SEAN.txt