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
# Characteristic path length
# Density | von Luca
# Average clustering coefficient and distribution of clustering coefficient
# Distribution of Cohesiveness
# Edge persistence under greedy attack (removing nodes with highest node degree first)
# Resilience/Survivability against random and targeted attacks (greedy attack)

print("Gnutella09 density: ", input.computed_density())
print("Gnutella09 average path length: ", input.compute_average_path_length())
