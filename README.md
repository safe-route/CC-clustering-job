# IMPORTANT THIS CODE HAS BEEN REFACTORED TO clustering-from-bq

# Clustering Job

For clustering from source (.csv of crime data) and the write it to JSON. Potentially can be improved to be invoked automatically for seamless write and modification to database.

## Summary

Codes are taken directly from clustering in machine learning repository, with some minor changes which includes disregarding visualizations codes, added main function wrappers and refactoring some functions.

`main.py` contains the code, `requirements.txt` containing dependencies of the code generated with `pip freeze > requirements.txt`.

## Details

- naked: Naive implementation. all depedency is installed directly and then the file is invoked.
- wrapped: Naive implementation. All dependency is installed directly then using `pip freeze` to use the `STROUT` to generate requirements of the function. Function is already wrapped in main function.