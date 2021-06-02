import pygame
import random
from defs import *
from nnet import Nnet

class Bird():
    def __init__(self, gameDisplay): #initialize bird and its features
        self.gameDisplay = gameDisplay
        self.state = BIRD_ALIVE #Bird starts alive
        self.img = pygame.image.load(BIRD_FILENAME) #Load image of bird
        self.rect = self.img.get_rect() #Is used to find the position of bird
        self.speed = 0 #start with 0 speed
        self.time_lived = 0 #start with no time lived (This is for the AI to see if its doing well
        self.set_position(BIRD_START_X, BIRD_START_Y) #Startposition
        # Initialize the neural network class with the given parameters
        self.nnet = Nnet(NNET_INPUTS, NNET_HIDDEN, NNET_OUTPUTS)

    def set_position(self, x, y):
        self.rect.centerx = x
        self.rect.centery = y

    def move(self, dt): #When moving
        distance = 0 #empty distance
        new_speed = 0 #empty speed

        distance = (self.speed * dt) + (0.5 * GRAVITY * dt * dt) #s = ut + 0.5at^2
        new_speed = self.speed + (GRAVITY * dt) #v = u + at

        self.rect.centery += distance #Set the position of the bird on the y axis
        self.speed = new_speed #Set speed of the bird

        if self.rect.top < 0: #If the bird goes above the screen
            self.rect.top = 0 #stay in screen
            self.speed = 0 #set speed to 0

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
        self.gameDisplay.blit(self.img, self.rect) #Draw every frame

    def check_status(self, pipes): #Check if the bird still lives
        if self.rect.bottom > DISPLAY_H: #If it is below the screen
            self.state = BIRD_DEAD
        else: #if not
            self.check_hits(pipes) #check if it hit with the pipes

    def check_hits(self, pipes):
        for p in pipes: #for every pipe in the list
            if p.rect.colliderect(self.rect): #When the bird collides with a pipe
                self.state = BIRD_DEAD
                break

    def update(self, dt, pipes):
        if self.state == BIRD_ALIVE: #if alive, call all the actions below
            self.time_lived += dt
            self.move(dt)
            self.draw()
            self.check_status(pipes)
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
        for p in pipes:
            # If the pipe is an upper pipe, and closer then the closest pipe stored, and at the right side of the pipe image
            if p.pipe_type == PIPE_UPPER and p.rect.right < closest and p.rect.right > self.rect.left:
                # Update the closest pipe position
                closest = p.rect.right
                bottom_y = p.rect.bottom

        # Get the horizontal distance of the bird to the pipe
        horizontal_distance = closest - self.rect.centerx

        # Get the vertical distance of the bird to the pipe
        vertical_distance = (self.rect.centery) - (bottom_y + PIPE_GAP_SIZE / 2)

        # Define the input values
        inputs = [
            ((horizontal_distance / DISPLAY_W) * 0.99) + 0.01,
            (((vertical_distance + Y_SHIFT) / NORMALIZER) * 0.99) + 0.01
        ]

        # Return the input values
        return inputs

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
       for b in self.birds:
           # Update the position of the bird by adding the movement and pipe information
           b.update(dt, pipes)
           # Check if the bird is living
           if b.state == BIRD_ALIVE:
               # Count the living birds
               num_alive += 1

       # Return the number of living birds
       return num_alive

