import numpy as np

# Define the number of bits for each feature
num_bits_rmt = 8
num_bits_emotion = 5

# Define the ranges for each feature
rmt_min, rmt_max = 0, 100
emotion_min, emotion_max = 0, 30

# Example dataset (replace with your actual dataset)
dataset = np.array([[50, 28, 4, 26],
                    [70, 12, 8, 14],
                    [90, 10, 20, 30]])

def normalize_feature(value, min_val, max_val):
    # Scale the feature value to the range [0, 1]
    normalized = (value - min_val) / (max_val - min_val)
    return normalized

def convert_to_binary(value, num_bits):
    # Convert the normalized value into a binary representation
    binary_repr = np.binary_repr(int(value * (2**num_bits - 1)), width=num_bits)
    return binary_repr

# Convert each feature into binary chromosomes
binary_chromosomes = []
for data_point in dataset:
    rmt_normalized = normalize_feature(data_point[0], rmt_min, rmt_max)
    rmt_binary = convert_to_binary(rmt_normalized, num_bits_rmt)
    
    emotion_binary = ''
    for i in range(1, len(data_point)):
        emotion_normalized = normalize_feature(data_point[i], emotion_min, emotion_max)
        emotion_binary += convert_to_binary(emotion_normalized, num_bits_emotion)
    
    chromosome = rmt_binary + emotion_binary
    binary_chromosomes.append(chromosome)

# Print the binary chromosomes
for chromosome in binary_chromosomes:
    print(chromosome)
    
def calculate_fitness(chromosome):
    # Convert the binary chromosome back to decimal values
    decimal_values = [int(binary_value, 2) for binary_value in chromosome]
    
    # Calculate the fitness as the sum of decimal values
    fitness = sum(decimal_values)
    
    return fitness  

    
    # Define the population size
population_size = len(binary_chromosomes)

# Define the Boltzmann selection temperature
temperature = 1.0

# Define the number of individuals to be selected
num_selected = 2

# Perform Boltzmann selection
selected_chromosomes = []
for _ in range(num_selected):
    # Calculate the Boltzmann probabilities for each chromosome
    probabilities = []
    for chromosome in binary_chromosomes:
        fitness = calculate_fitness(chromosome)  # Replace with your fitness function
        prob = np.exp(fitness / temperature)
        probabilities.append(prob)
    
    # Normalize the probabilities to sum up to 1
    probabilities /= np.sum(probabilities)
    
    # Select an individual probabilistically based on Boltzmann probabilities
    selected_index = np.random.choice(range(population_size), p=probabilities)
    selected_chromosome = binary_chromosomes[selected_index]
    
    # Add the selected chromosome to the selected list
    selected_chromosomes.append(selected_chromosome)

# Print the selected chromosomes
for chromosome in selected_chromosomes:
    print(chromosome)
