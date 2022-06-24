"""
A set of functions generally useful for various components of the project.
"""




def print_dict(a_dict : dict):
    """
    Prints a dictionary in a human readable format.
    
    Parameters:
        a_dict(dict): The dictionary to print.
    """
    for item in a_dict.items():
        print(item[0])
        print(item[1])
        print()