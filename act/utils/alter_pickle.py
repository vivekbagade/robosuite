import pickle
import os
import ast
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('file_path', None, 'Path to the input pickle file.')
flags.DEFINE_string('key', None, 'The key to add or update in the dictionary.')
flags.DEFINE_string('value', None, 'The value for the key. This will be evaluated to a Python type.\n'
                                  'Examples:\n'
                                  "'some_string' (for a string)\n"
                                  "123 (for an integer)\n"
                                  "3.14 (for a float)\n"
                                  "True (for a boolean)\n"
                                  "'[1, 2, 3]' (for a list)")
flags.DEFINE_string('output_path', None, 'Path for the output pickle file. If not provided, the original file will be overwritten.')

flags.mark_flag_as_required('file_path')
flags.mark_flag_as_required('key')
flags.mark_flag_as_required('value')


def modify_pickle(file_path, key, value_str, output_path=None):
    """
    Loads a dictionary from a pickle file, adds a key-value pair,
    and saves it back.

    Args:
        file_path (str): Path to the input pickle file.
        key (str): The key to add to the dictionary.
        value_str (str): The string representation of the value to add.
                         It will be evaluated to its Python type.
        output_path (str, optional): Path to save the modified pickle file.
                                     If None, overwrites the original file.
                                     Defaults to None.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # Load the dictionary from the pickle file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        print(f"Error: The data in {file_path} is not a dictionary.")
        return

    # Safely evaluate the value string to its actual type
    try:
        value = ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        # If literal_eval fails, treat it as a plain string.
        value = value_str

    # Add or update the field
    print(f"Original data: {data}")
    data[key] = value
    print(f"Updated data:  {data}")


    # Determine the output path
    if output_path is None:
        output_path = file_path

    # Save the modified dictionary back to a pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Successfully modified pickle file and saved to {output_path}")

def main(argv):
    """Main function."""
    del argv  # Unused.
    modify_pickle(FLAGS.file_path, FLAGS.key, FLAGS.value, FLAGS.output_path)

if __name__ == "__main__":
    app.run(main)