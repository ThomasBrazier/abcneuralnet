# recipe/bld.bat
"%R%" CMD INSTALL --build .

IF %ERRORLEVEL% NEQ 0 EXIT /B %ERRORLEVEL%