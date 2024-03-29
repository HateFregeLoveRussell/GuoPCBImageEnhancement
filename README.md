Image Enhancement Algorithm for PCB Fault Detection
From paper DOI 10.1109/ISIE.2011.54

## How to Run
To run this project, the following steps must be taken:
1. Clone the project:
```
git clone https://github.com/HateFregeLoveRussell/GuoPCBImageEnhancement.git
```
2. Change into the project directory:
```
cd GuoPCBImageEnhancement
```
3. Create a virtual environment:
```
python3 -m venv guo
```
4. Activate the virtual environment:

For Linux:
  ```
  source guo/bin/activate.csh
  ```
For Windows:
  ```
  source guo/Scripts/activate
  ```
5. Install dependicies:
```
pip install -r requirements.txt
```
6. Run the project:
```
python GuoMethod.py /home/a7shahba/GuoPCBImageEnhancement/Inputs /home/a7shahba/GuoPCBImageEnhancement/tmp 0 3
```
Ensure the folder for the input folder and save folder exist. If the running on Windows, ensure that the file path has `\\`. Index starts at 0 and ends at one after the last index. For example, if we are processing 5 images, to process all images, the indexs are 0 6.
