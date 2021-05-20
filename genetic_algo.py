import torch
import pygad 
import pygad.torchga as torchga

class Genetic_algo():
    '''aa'''
    def __init__(self,num_generations = 250, output_size=1):
        #super().__init__()
            # Create the PyTorch model.
        self.input_layer = torch.nn.Linear(3, 50)
        self.relu_layer = torch.nn.ReLU()
        self.output_layer = torch.nn.Linear(50, output_size)
        self.model = torch.nn.Sequential(self.input_layer,
                                self.relu_layer,
                                self.output_layer)
            # print(model)
            # Create an instance of the pygad.torchga.TorchGA class to build the initial population.
        self.torch_ga = torchga.TorchGA(model=self.model,
                            num_solutions=10)

        self.loss_function = torch.nn.L1Loss()

            # Data inputs
        self.data_inputs = torch.tensor([[1.0, 1.0, 1.0],
                                [2.0,2.0,2.0],
                                [3.0,3.0,3.0],
                                [4.0,4.0,4.0]])

            # Data outputs
        self.data_outputs = torch.tensor([[0.0,0.0],
                                [0.0,1.0],
                                [1.0,0.0],
                                [1.0,1.0]])
            # Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
        self.num_generations = num_generations # Number of generations.
        self.num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
        self.initial_population = self.torch_ga.population_weights # Initial population of network weights
        self.parent_selection_type = "sss" # Type of parent selection.
        self. crossover_type = "single_point" # Type of the crossover operator.
        self.mutation_type = "random" # Type of the mutation operator.
        self.mutation_percent_genes = 10 # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
        self.keep_parents = -1 # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

        self.ga_instance = pygad.GA(num_generations=self.num_generations, 
                        num_parents_mating=self.num_parents_mating, 
                        initial_population=self.initial_population,
                        fitness_func=self.fitness_func, #pygad.py changed line 535 ...== 3 "
                        parent_selection_type=self.parent_selection_type,
                        crossover_type=self.crossover_type,
                        mutation_type=self.mutation_type,
                        mutation_percent_genes=self.mutation_percent_genes,
                        keep_parents=self.keep_parents,
                        on_generation=None)#self.callback_generation)#pygad.py changed line 629,647  ...=2



    
    def fitness_func(self,solution, sol_idx):
        #global data_inputs, data_outputs, torch_ga, model, loss_function

        model_weights_dict = torchga.model_weights_as_dict(model=self.model,
                                                        weights_vector=solution)

        # Use the current solution as the model parameters.
        self.model.load_state_dict(model_weights_dict)

        predictions = self.model(self.data_inputs)
        abs_error = self.loss_function(predictions, self.data_outputs).detach().numpy() + 0.00000001

        solution_fitness = 1.0 / abs_error

        return solution_fitness
    
    def callback_generation(self,ga_instance):
        print("Generation = {generation}".format(generation=ga_instance.generations_completed))
        print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))



    def set_data(self,data_input,data_output):
        self.data_input=torch.tensor([data_input])
        self.data_output=torch.tensor([data_output])

    def learn(self):
        '''aaa '''
        self.ga_instance.run()
        # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
        #self.ga_instance.plot_result(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

    def best_solution(self):
        # Returning the details of the best solution.
        solution, solution_fitness, solution_idx = self.ga_instance.best_solution()
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
        print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))


        # Fetch the parameters of the best solution.
        best_solution_weights = torchga.model_weights_as_dict(model=self.model,
                                                        weights_vector=solution)
        self.model.load_state_dict(best_solution_weights)
        predictions = self.model(self.data_inputs)
        print("Predictions : \n", predictions.detach().numpy())

        #abs_error = self.loss_function(predictions, self.data_outputs)
        #print("Absolute Error : ", abs_error.detach().numpy())

        return predictions

    def predict(self,data):
        data= torch.tensor(data)
        best_solution_weights = torchga.model_weights_as_dict(model=self.model,
                                                        weights_vector=solution)
        self.model.load_state_dict(best_solution_weights)
        predictions = self.model(data)
        print("Predictions : \n", predictions.detach().numpy())
        return predictions