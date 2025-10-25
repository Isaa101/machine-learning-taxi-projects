#######################
#    map_loader.py    #
#######################
# Author: Enrique Mateos Melero
# File to prepare map configurations for OpenAi Gym environment


def prepare_for_env(filepath):
    print("[INFO]: Preparing map for env")
    try:
        print("[INFO]: Loading map data from file")
        with open(filepath, "r") as map_data:
            mapping = map_data.readlines()
    
    except FileNotFoundError:
        raise Exception("Specified file does not exist")
    
    for i,  row in enumerate(mapping):
        new_row = row.replace("\n", "")
        mapping[i] = new_row
    print("[DEBUG]: Map prepared for env")
    return mapping


if __name__ == "__main__":
    prepare_for_env("./maps/map_1.txt")