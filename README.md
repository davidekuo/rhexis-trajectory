# 👁 Capsulorrhexis Trajectories for Automated Surgical Training Feedback 🥼

## Contributors 💻
David Kuo MD, Ben Viggiano, Riya Sinha, Ben Ehlert

## Gettting Started ⚡️
1) Clone the repository to your computer.<br>  

2) Run the following command from inside the repository to set up the conda environment: <br>  `conda env create -f environment.yml` <br>  

3) Download our dataset from the following Google Drive link:
`https://drive.google.com/drive/folders/1CEiRaF4CoqUOh7OzYxVoQGf39p3Hq2Qz?usp=sharing`

    Save the dataset on your computer and get it's full filepath for the next step.

4) Update the first line of the file `rhexis_config.txt` so that the first line contains:
`DATA_LOCATION:<location>` <br>  where `<location>` is the full file path to the location of of the folder 'datasets'. (These file's changes are not tracked, so others using the repository can also specify their own paths)

5) You are good to go! Check out `Detectron2_Rhexis.ipynb` 🙂
