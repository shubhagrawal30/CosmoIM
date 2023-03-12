import os

from simim._paths import _paths

class create_paths():
    """Does all of the initialization"""

    def __init__(self):

        path = _paths()

        print("Please specify a path to save data directories.")
        print("Specifying no path will set the path to your home directory.")
        root = input("Path: ")
        if root == '':
            root = '~'
        elif not os.path.exists(root):
            raise NameError("Specified path does not exist. Please create path and try again.")
        path._setuppath(root=root)

if __name__ == '__main__':
    create_paths()