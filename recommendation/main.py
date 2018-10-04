"""
In order to run this just download spark and extract somewhere.

Then use command below:

/path/to/your/spark/bin/spark-submit ./main.py cross_valid.py metrics.py --your-options.

"""


from cross_valid import ALSTrainFlow
import tensorflow as tf

tf.convert_to_tensor
als_flow = ALSTrainFlow(rating_path='/home/vigen/Documents/job/recomm_data/UserEvents.csv',
                        user_path='/home/vigen/Documents/job/recomm_data/Users.csv',
                        book_path='/home/vigen/Documents/job/recomm_data/Books.csv')

als_flow.read_data()

best_model = als_flow.cross_validate_model(ranks=[20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 90],
                                           regParams=[0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
                                           test_portion=0.1)


als_flow.create_rating_matrix(best_model, path_to_save='./rating.csv')


