# Clustering Job

For clustering from source (.csv of crime data) and the write it to JSON. Potentially can be improved to be invoked automatically for seamless write and modification to database.

## Details
- naked: Naive implementation. all depedency is installed directly and then the file is invoked.
- wrapped: Naive implementation. All dependency is installed directly then using `pip freeze` to use the `STROUT` to generate requirements of the function. Function is already wrapped in main function.