import os
from GraphInput import TxtFileGraphInput

script_dir = os.path.dirname(os.path.realpath(__file__))
gnutella_file_path = os.path.join(
    script_dir, "../data/p2p-Gnutella09/p2p-Gnutella09.txt"
)

input = TxtFileGraphInput()
input.read_input_file_and_convert(gnutella_file_path, directed=True)

# Implemented or reserved by Luca
print("Gnutella09 average path length: ", input.compute_average_path_length())
print("Gnutella09 density: ", input.computed_density())
# print("Gnutella09 Edge persistence under greedy attack: ", input.todo())


# Implemented or reserved by Jan

(
    average_clustering_coefficient,
    clustering_coefficient_distribution,
) = input.compute_clustering_coefficient()

print("Gnutella09 Average clustering coefficient: ", average_clustering_coefficient)
print(
    "Gnutella09 Clustering coefficient distribution: ",
    clustering_coefficient_distribution,
)


# Such dir was aus Luca :)
# print("Gnutella09 Distribution of cohesiveness: ", input.todo())
# print("Gnutella09 Resilience/Survivability: ", input.todo())
