This code is to generate original features for SDM-VQA based on the previous work (FAST, 2019-TMM).

To run the code, OpenCV is requested and in current version, we used OpenCV 2.4.13. 

- Set the proper PATH for OpenCV in `CompileMex.m` and Compile the `.cpp` files;
- Run `demo_universal.m` for you purpose by changing the file path.

NOTE: The generated data is `.mat` for each video file (in `./iFAST_key_10`). For an easy usage in Python, we convert the files to `.pkl` with `pickle` within the database (in `./data/`).

If any question, feel free to contact me.
