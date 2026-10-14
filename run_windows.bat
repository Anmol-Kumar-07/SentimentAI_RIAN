@echo off
cd /d "%~dp0"
if not exist .venv (
  py -m venv .venv
  if errorlevel 1 goto error
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 goto error
if not exist sentiment_3class_logistic_regression_model.pkl (
  echo Sentiment model files are missing. Training models from labeled Amazon reviews; internet is required.
  python train_models.py
  if errorlevel 1 goto error
)
python app.py
pause
goto end
:error
echo Setup or training failed. Read the error above and check internet connectivity.
pause
:end
