1 - The Data analysis notebook expects the csv files to be in a folder called "book-recommend-data".

2 - A default anaconda installation should be able to run the code without any problems, however, I
added a requirements.txt to be used with pip

3 - The result matrix is 1263x281. That's because I'm using only the books that sold at least 10 times, 
and I'm also using the users that aren't on the Users.csv file but are on the UserEvents.csv file