import pygame
import random
from defs import *
from nnet import Nnet
import numpy as np


class Bird:
    """ Object to store the information of bird movements
    """

    def __init__(self, gameDisplay):
        """ Initialize the bird movement class

        :param gameDisplay: pygame window size
        """

        # Set the display width and height
        self.gameDisplay = gameDisplay

        # Set the bird on living
        self.state = BIRD_ALIVE

        # Load the image of the bird
        self.img = pygame.image.load(BIRD_FILENAME)

        # Get the position of the bird image
        self.rect = self.img.get_rect()

        # Set the start speed at 0
        self.speed = 0

        # Set the start living time at 0 (This is for the AI to see if it works well)
        self.time_lived = 0

        # Set the start position of the bird
        self.set_position(BIRD_START_X, BIRD_START_Y)

        # Initialize the neural network class with the given parameters
        self.nnet = Nnet(NNET_INPUTS, NNET_HIDDEN, NNET_OUTPUTS)

        # Set the bird fitness at 0
        self.fitness = 0

    def set_position(self, x, y):
        """ Function to get the initially position of the bird

        :param x: integer, x-position of the bird
        :param y: integer, y-positiion of the bird
        """
        
        self.rect.centerx = x
        self.rect.centery = y

    def move(self, dt):
        """  Function to update the bird movement variables

        :param dt: integer, time movement in milliseconds
        """

        # When moving, restart the distance and speed
        distance = 0
        new_speed = 0

        # Calculate the distance with s = ut + 0.5at^2
        distance = (self.speed * dt) + (0.5 * GRAVITY * dt * dt)

        # Calculate the speed with v = u + at
        new_speed = self.speed + (GRAVITY * dt)

        # Set the position of the bird on the y axis
        self.rect.centery += distance

        # Set speed of the bird
        self.speed = new_speed

        # If the bird goes above the screen
        if self.rect.top < 0:
            # Let the bird stay in screen
            self.rect.top = 0
            # Set the speed to 0
            self.speed = 0

    def jump(self, pipes):
        """ Function to decide if the bird should flap or not

        :param pipes: list, containing sets of pipe objects
        """

        # Get the input values for the neural network
        inputs = self.get_inputs(pipes)

        # Get the maximal output value of the neural network
        val = self.nnet.get_max_value(inputs)

        # Decide if the bird should flap
        if val > JUMP_CHANCE:
            # Flap the bird when the output value is higher than the set threshold
            self.speed = BIRD_START_SPEED

    def draw(self):
        """ Function to visualize the bird
        """

        # Draw every frame
        self.gameDisplay.blit(self.img, self.rect)

    def check_status(self, pipes):
        """ Function to check if the bird is still alive

        :param pipes: list, containing pipe information
        """

        # If the bird is below the screen
        if self.rect.bottom > DISPLAY_H:
            # Set the bird status as dead
            self.state = BIRD_DEAD
        # If the bird is still at the screen
        else:
            # Check if the bird hit a pipe
            self.check_hits(pipes)

    def check_hits(self, pipes):
        """ Function to check if the bird hits a pipe

        :param pipes: list, containing pipe information
        """

        # For every pipe in the list
        for pipe in pipes:
            # When the bird collides with a pipe
            if pipe.rect.colliderect(self.rect):
                # Set the bird status as dead and stop the loop
                self.state = BIRD_DEAD
                break

    def update(self, dt, pipes):
        """ Function to update the position of a living bird

        :param dt: integer, time movement in milliseconds
        :param pipes: list, containing pipe information
        """

        # Check if the bird is living
        if self.state == BIRD_ALIVE:
            # Update the living time of the bird
            self.time_lived += dt
            # Update the position of the bird
            self.move(dt)
            # Draw the bird at the new position
            self.draw()
            # Check if the bird hits a pipe with the new position
            self.check_status(pipes)
            # Check if the bird needs to flap
            self.jump(pipes)

    def get_inputs(self, pipes):
        """ Function to get the input values for the first layer, input layer

        :param pipes: list, containing sets of pipe objects
        :return:
            inputs: list, containing floats as input values for the neural network
        """

        # Initialize the closest x-position of a pipe with a too big value
        closest = DISPLAY_W * 2

        # Initialize the y-position of a pipe
        bottom_y = 0

        # Check which pipe (upper or lower) is the closest
        for pipe in pipes:
            # If the pipe is an upper pipe, and closer then the closest pipe stored, and at the right side of the pipe image
            if pipe.pipe_type == PIPE_UPPER and pipe.rect.right < closest and pipe.rect.right > self.rect.left:
                # Update the closest pipe position
                closest = pipe.rect.right
                bottom_y = pipe.rect.bottom

        # Get the horizontal distance of the bird to the pipe
        horizontal_distance = closest - self.rect.centerx

        # Get the vertical distance of the bird to the pipe
        vertical_distance = self.rect.centery - (bottom_y + PIPE_GAP_SIZE / 2)

        # Define the input values
        inputs = [
            ((horizontal_distance / DISPLAY_W) * 0.99) + 0.01,
            (((vertical_distance + Y_SHIFT) / NORMALIZER) * 0.99) + 0.01
        ]

        # Return the input values
        return inputs

    def create_offspring(p1, p2, gameDisplay):
        """ Function to create a child bird from two parent birds based on their neural net weights

        :param p1: numpy array, 2-dimensional containing floats numbers representing weights
        :param p2: numpy array, 2-dimensional containing floats numbers representing weights
        :param gameDisplay: pygame window size
        :return:
            bird: class, containg information of the bird
        """

        # Create a new bird
        new_bird = Bird(gameDisplay)

        # Give the child bird a defined set of weights
        new_bird.nnet.create_mixed_weights(p1.nnet, p2.nnet)

        # Return the child bird
        return new_bird

    def reset(self):
        """ Function to reset the bird and its properties
        """

        # Set all bird properties to the defined starting values
        self.state = BIRD_ALIVE
        self.speed = 0
        self.fitness = 0
        self.time_lived = 0
        self.set_position(BIRD_START_X, BIRD_START_Y)


class BirdCollection:
    """ Object to store information of the bird collection
       """

    def __init__(self, gameDisplay):
        """ Initialize the bird collection class

        :param gameDisplay: pygame window size
        """

        # Set the display width and height
        self.gameDisplay = gameDisplay

        # Store created birds
        self.birds = []

        # When the class is initialized, an new generation of birds will be created
        self.create_new_generation()

    def create_new_generation(self):
        """ Function to create a new generation of birds
        """

        # Start with an empty bird list
        self.birds = []

        # Add as many birds as the defined population size
        for i in range(0, GENERATION_SIZE):
            # Extend the bird list with the new birds
            self.birds.append(Bird(self.gameDisplay))

    def update(self, dt, pipes):
        """

        :param dt: integer, time movement in milliseconds
        :param pipes: list, containing pipe information
        :return:
            num_alive: integer, number of living birds
        """

        # Start the counter of alive birds at 0
        num_alive = 0

        # Check every bird in the bird list
        for bird in self.birds:
            # Update the position of the bird by adding the movement and pipe information
            bird.update(dt, pipes)
            # Check if the bird is living
            if bird.state == BIRD_ALIVE:
                # Count the living birds
                num_alive += 1

        # Return the number of living birds
        return num_alive

    def evolve_population(self):
        """ Function to evolve the birds each generation
        """

        # For every bird in the list
        for bird in self.birds:
            # Update the bird fitness determined by living time and the distance to the gap of the pipes
            bird.fitness += bird.time_lived * PIPE_SPEED

        # Sort the birds by their fitness
        self.birds.sort(key=lambda x: x.fitness, reverse=True)

        # Set the cut off value
        cut_off = int(len(self.birds) * MUTATION_CUT_OFF)

        # Set the good birds which are the top performing birds
        good_birds = self.birds[0:cut_off]

        # Set the bad birds, which is the rest
        bad_birds = self.birds[cut_off:]

        # Set the number of bad birds that should be kept
        num_bad_to_take = int(len(self.birds) * MUTATION_BAD_TO_KEEP)

        # For every bad bird
        for bird in bad_birds:
            # Modify the neural network weights
            bird.nnet.modify_weight()

        # Create a list of new birds that are needed for every deleted bad bird
        new_birds = []

        # Pick the birds to take to the next round
        idx_bad_to_take = np.random.choice(np.arange(len(bad_birds)), num_bad_to_take, replace=False)

        # Add the birds to take to the next round to the bird list
        for index in idx_bad_to_take:
            new_birds.append(bad_birds[index])

            # Extend the new bird list with the good birds to keep
        new_birds.extend(good_birds)

        # When more birds are needed
        while len(new_birds) < len(self.birds):

            # Pick random parents from the good birds
            idx_to_breed = np.random.choice(np.arange(len(good_birds)), 2, replace=False)

            # Make sure the parent birds are not the same
            if idx_to_breed[0] != idx_to_breed[1]:

                # Create a offspring bird from the two parent birds
                new_bird = Bird.create_offspring(good_birds[idx_to_breed[0]], good_birds[idx_to_breed[1]],
                                                 self.gameDisplay)

                # There is a chance that the weights of the neural network are changed
                if random.random() < MUTATION_MODIFY_CHANCE_LIMIT:
                    new_bird.nnet.modify_weight()

                # Add the created bird to the list
                new_birds.append(new_bird)

        # Reset the birds for every bird in the new list
        for bird in new_birds:
            bird.reset()

        # Update the bird list
        self.birds = new_birds
