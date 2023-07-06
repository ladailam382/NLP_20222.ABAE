import pickle
import torch

def convert_txt_to_pkl(txt_file, pkl_file):
  """Converts a TXT file to a PKL file.

  Args:
    txt_file: The path to the TXT file to convert.
    pkl_file: The path to the PKL file to save the converted data to.

  """

  with open(txt_file, "r") as f:
    data = f.read()

  torch.save([[data]], pkl_file)

if __name__ == "__main__":
  txt_file = "Data/test_input.txt"
  pkl_file = "data.zip"

  convert_txt_to_pkl(txt_file, pkl_file)