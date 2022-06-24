"""
A set of functions generally useful for various components of the project.
"""

def print_dict_list(dict_list:list):
    """
    Prints a list of dictionaries in a human readable format.
    
    Parameters:
        dict_list(list): The list of dictionaries to print.
    """
    for item in dict_list:
        print_dict(item)
        print()


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