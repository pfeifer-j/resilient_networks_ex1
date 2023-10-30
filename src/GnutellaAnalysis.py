from exercise_1.src.GraphInput import TxtFileGraphInput

input = TxtFileGraphInput()
gnutella_file_path = "exercise_1/data/p2p-Gnutella09/p2p-Gnutella09.txt"
input.read_input_file_and_convert(gnutella_file_path, directed=True)
