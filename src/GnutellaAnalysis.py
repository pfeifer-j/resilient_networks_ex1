import os
from GraphInput import TxtFileGraphInput

script_dir = os.path.dirname(os.path.realpath(__file__))
gnutella_file_path = os.path.join(
    script_dir, "../data/p2p-Gnutella09/p2p-Gnutella09.txt"
)

input = TxtFileGraphInput()
input.read_input_file_and_convert(gnutella_file_path, directed=True)

# Calculations:
# Power-law properties
# Characteristic path length | *von Luca implementiert*
# Density | *von Luca implementiert*
# Average clustering coefficient
# Distribution of clustering coefficient
# Distribution of Cohesiveness
# Edge persistence under greedy attack (removing nodes with highest node degree first)
# Resilience/Survivability against random and targeted attacks (greedy attack)

# Done
# print("Gnutella09 average path length: ", input.compute_average_path_length())
# print("Gnutella09 density: ", input.computed_density())

# Todo
print(
    "Gnutella09 Average clustering coefficient: ",
    input.compute_average_clustering_coefficient(),
)
# print("Gnutella09 Distribution of clustering coefficient: ", input.todo())
# print("Gnutella09 Distribution of cohesiveness: ", input.todo())
# print("Gnutella09 Edge persistence under greedy attack: ", input.todo())
# print("Gnutella09 Resilience/Survivability: ", input.todo())
