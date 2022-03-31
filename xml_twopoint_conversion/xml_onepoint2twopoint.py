# Quick script to convert the xml files from using one point to two points in all cases

input_file = "test_cvat_annotations.xml"


def convert_to_two_points(line :str):
    """
    A function that reads in the line containing the point values and alters the line to contain
    two points instead of one if necessary.

    Parameters:
        line: String containing the line of the xml file that contains point values
    
    Returns:
        String with two instances of points
        - If original string had two instances of point values, the string will not be changed
        - If original string had one instance of point values, the string will be altered
    """

    # Find where the point variable starts
    point_index = line.find('points="')

    # Save the rest of the string in a split
    remaining_str = line[point_index:].split(" ")

    # Find the start and end index of the point values
    point_value_start = remaining_str[0].find('"') +1
    point_value_end = remaining_str[0].find('"',point_value_start+1)

    # Collect the point value containing string
    point_values = remaining_str[0][point_value_start:point_value_end]

    # If string does not contain the ; seperator then add the duplicate point values
    if ";" not in point_values:
        new_line = line[0:point_index]
        new_line = new_line + 'points="'
        new_line = new_line + point_values
        new_line = new_line + ';'
        new_line = new_line + point_values + '"'
        new_line = new_line + " " + remaining_str[1]
        line = new_line

    return line





newtext = ""
with open(input_file) as file:
     # Read in lines from the file
    lines = file.readlines()

    # For each line
    for line in lines:

        # if the line contains the "point=", run the line through the function
        if "points=" in line:
            line = convert_to_two_points(line)
        
        # Add the line to the new text
        newtext = newtext + line


# Now write the newtext variable to an output file
with open("CONVERTED"+input_file,"w") as file:
    file.write(newtext)

