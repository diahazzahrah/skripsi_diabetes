SKRIPSI
- DIAH SITI FATIMAH AZZAHRAH
- NIM. 4611419056
- PROGRAM STUDI TEKNIK INFORMATIKA 2019

CARA MENJALANKAN SISTEM
- pastikan sudah menginstall python v.3

1. Extract rar. (skripsi_diabetes)
2. Open project in visual studio code
3. Begin a new virtual environment with Python 3 and activate it. (Project using 3.10)
- Jika venv tidak bisa di aktifkan (https://www.stanleyulili.com/powershell/solution-to-running-scripts-is-disabled-on-this-system-error-on-powershell/) 
4. Install the required packages using (or install per library)
     `pip install -r requirements.txt`
5. Using pip3 install --upgrade numpy==1.20.3
6. If tensorflow install is detected, it is uninstalled. (pip uninstall tensorflow)
8. Check version pip. (must 23.0.1)
7. Execute the command/terminal:
     `python app.py`



Install manual (after activate venv)
- pip install Flask 
- pip install numpy (versi 1.20.3)
- pip install neupy 
- pip install pandas
- pip install scikit-learn 

NOTE (when program can't running)
- Follow step for msys2 https://www.msys2.org/ (Try it first without following this step)
- Install mingw https://www.youtube.com/watch?v=Zcy981HhGw0 -> tutorial (Try it first without following this step) 
- install git https://gitforwindows.org/ -> MINGW64
- How to install https://phoenixnap.com/kb/how-to-install-git-windows 