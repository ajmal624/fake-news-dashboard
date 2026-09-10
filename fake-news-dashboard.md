# fake-news-dashboard

Unzip the File
use internet for all installation

Install Python 3.14, then verify the installation:
    Press Windows + R, type `cmd`, and press Enter.
    In Command Prompt, run:
        python --version

Open Visual Studio Code and install the required Python extensions.

Go to File → New Window → Open Folder, then select the `fake-news-dashboard` folder.

Open the terminal and run:
    py -m venv .venv
    venv\Scripts\Activate
    pip install -r requirements.txt

Run the application:
    streamlit run app.py

On the output page, Enter the News that are available in fake-news-dashboard → data → raw → Fake.csv & True.csv (or) example input.txt and run the prediction to view the result.