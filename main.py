import random
import numpy as np


def W_generator(hideen_neurons: int, sequence_len: int):
    matrix_W1 = []
    for neuron in range(hideen_neurons):
        temp_list = []
        for digit in range(sequence_len):
            temp_list.append(random.uniform(-1, 1))
        matrix_W1.append(temp_list)
    return np.array(matrix_W1)


def generate_context_neurons(amount:int):
    context_neurons=[]
    neuron=[0]
    for i in range (amount):
        context_neurons.append(neuron)
    return context_neurons


def generate_T(hidden_neurons: int):
    matrix = []
    for neuron in range(hidden_neurons):
        temp_list = []
        for weight in range(1):
            temp_list.append(random.uniform(-1, 1))
        matrix.append(temp_list)
    return np.array(matrix)


def activation_function(hidden_layer_matrix):
    for i in range(len(hidden_layer_matrix)):
        for j in range(len(hidden_layer_matrix[i])):
            if hidden_layer_matrix[i][j] < 0:
                hidden_layer_matrix[i][j] = hidden_layer_matrix[i][j] * 0.01
    return hidden_layer_matrix


def activation_function_derivative(before_activ_matrix):
    for i in range(len(before_activ_matrix)):
        for j in range(len(before_activ_matrix[i])):
            if before_activ_matrix[i][j] < 0:
                before_activ_matrix[i][j] = 0.01
            else:
                before_activ_matrix[i][j] = 1
    return before_activ_matrix


def calculate_error(sample, output):
    temp_error = pow(sample, 2) + pow(output, 2)
    temp_error = (temp_error - 2 * sample * output)/2
    return temp_error


def W1_update(W1_before, ratio, output, sample, W2_after, function, start_neurons):
    temp_error = output - sample
    temp_W1 = ratio * temp_error
    temp_W1 = W2_after * temp_W1
    temp = function @ start_neurons.T
    temp_W1 = temp_W1 @ temp
    temp_W1 = W1_before - temp_W1
    return temp_W1


def W2_update(W2_before, ratio, output, sample, hidden_neurons):
    temp_error = output - sample
    temp_W2 = ratio * temp_error
    temp_W2 = hidden_neurons * temp_W2
    temp_W2 = W2_before - temp_W2.T
    return temp_W2


def W3_update(W3_before, ratio, output, sample, W2_after, function):
    temp_error = output - sample
    temp_W3 = ratio * temp_error
    temp_W3 = W2_after * temp_W3
    temp = function * output
    temp_W3 = temp_W3 @ temp
    temp_W3 = W3_before - temp_W3
    return temp_W3


def T_update(T, output, sample):
    return T + (output - sample)


def update_hidden_layer(hidden,context_neurons,T):
    hidden = hidden + context_neurons - T
    return hidden


def print_info(i,count,sequence,sample,error,output):
    print(f"Step - {count}")
    print(f"Sequence= {sequence[i]} : Sequence+1 = {sample[i]}")
    print(f"Error= {error} : Guessed num: {output[0][0]}")


if __name__ == '__main__':
    # 1 3 5 7 9 11 13
    sequence = [[1, 3, 5],[3, 5, 7],[5, 7, 9],[7, 9, 11]]
    sample = [7, 9, 11, 13]
    output_neurons = int(input("Type in number of digits to guess: "))
    learning_koef = float(input("Type in learning koef: "))
    num_of_neurons = output_neurons + len(sequence[0])
    context_neurons = generate_context_neurons(num_of_neurons)
    W1_matrix = W_generator(num_of_neurons, len(sequence[0]))
    W2_matrix = W_generator(output_neurons, num_of_neurons)
    W3_matrix = W_generator(num_of_neurons, num_of_neurons)
    T = generate_T(num_of_neurons)
    steps = 0
    while steps < 1000:
        for i in range(len(sequence)):
            steps += 1
            start_neurons = np.array([sequence[i]]).T
            context_neurons = W3_matrix @ context_neurons
            hidden_layer_neurons = W1_matrix @ start_neurons
            hidden_layer_neurons = update_hidden_layer(hidden_layer_neurons,context_neurons,T)
            before_activation = hidden_layer_neurons
            hidden_layer_neurons = activation_function(hidden_layer_neurons)
            context_neurons = hidden_layer_neurons
            output = W2_matrix @ hidden_layer_neurons
            error = abs(calculate_error(sample[i], context_neurons[0][0]))
            print_info(i, steps, sequence, sample, error, output)
            W2_matrix = W2_update(W2_matrix, learning_koef, output[0][0], sample[i], hidden_layer_neurons)
            W1_matrix = W1_update(W1_matrix, learning_koef, output[0][0], sample[i], W2_matrix, activation_function_derivative(before_activation), start_neurons)
            W3_matrix = W3_update(W3_matrix, learning_koef, output[0][0], sample[i], W2_matrix, activation_function_derivative(before_activation))
            T = T_update(T, output[0][0], sample[i])