import glob

def get_all_files_in_directory(directory, extension=''):
    '''
    A function used to recursivly extract all files with the given extension from a directory.

    :param directory: The directory we want to extract the files from.
    :param extension: The extension we want to use. All files are retrieved on default.

    :returns: The files in the directory with the given extension.
    '''
    if directory[-1] == '/':
        directory = directory[:-1]
    return glob.glob(directory + '/**/*' + extension, recursive=True)

def get_song_number_for_filename(filename):
    '''
    A function return the song number for the given filename.
    Format normally is <genre>.xxxxx.au
    '''
    split_name = filename.split('.')
    if len(split_name) == 3:
        return int(split_name[1])
    else:
        return -1