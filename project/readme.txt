Install latest version of Anaconda 
Then Form Anaconda Navigator create Python New Environment with python 3.7 ->  Intsall latest version of Tensorflow Nltk Flask Pickle5 scikit-learn Pandas from Anaconda prompt.


In Anaconda Terminal:

conda install -c anaconda colorama

python
>>> from project import db, create_app
>>> db.create_all(app=create_app())

set FLASK_APP=project
set FLASK_DEBUG=1
flask run