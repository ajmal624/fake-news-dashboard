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

## ⚠️ Important Files – Large File Limitation

Some important dataset files used by this project could not be uploaded
to this GitHub repository because of GitHub's file-size limitations.

These files are required for training and reproducing the complete
Fake News Detection project and are kept separately.

### Important files not included in this repository

- `data/raw/Fake.csv`
- `data/raw/True.csv`

These dataset files contain the real and fake news data used for
training and evaluating the Fake News Detection model.

The files are **not deleted or unnecessary**; they are excluded from
GitHub only because of their large file sizes.

The trained model (`models/fake_news_model.pt`), vocabulary
(`models/vocab.pkl`), application code (`app.py`), and training code
(`train.py`) are included in this repository.

The model can be retrained using `train.py` when the required
`Fake.csv` and `True.csv` datasets are placed inside the `data/raw/`
directory.

> **Note:** The large dataset files can be provided separately if required.
