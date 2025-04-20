# Définition des variables d’environnement (correct pour Flask)
$env:FLASK_APP = "src.app"
$env:FLASK_ENV = "development"
$env:FLASK_DEBUG = 0

# Affichage dans le style que tu veux
Write-Host "FLASK_APP = src/app.py"
Write-Host "FLASK_ENV = development"
Write-Host "FLASK_DEBUG = 0"
Write-Host "In folder $((Get-Location).Path)"
