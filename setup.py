from cx_Freeze import setup, Executable

setup(name="Basic object detection software", version="0.1", description="This software detects objects real time",
      executables=[Executable("main.py")])
