# dendritic-spine-shape-analysis
## Install
1. Download code
2. Unzip [CGAL files](https://github.com/pv6/cgal-swig-bindings/releases/download/python-build/CGAL.zip) next to code, e.g. `PATH_TO_CODE\CGAL\...`
3. Install [Anaconda](https://www.anaconda.com/)
4. Open Anaconda
5. Execute
```cmd
cd PATH_TO_CODE
conda create --name spine-analysis -c conda-forge --file requirements.txt -y
```
4. Copy CGAL 
## Run
1. Open Anaconda
2. Execute
```cmd
cd PATH_TO_CODE
conda activate spine-analysis
jupyter notebook
```
