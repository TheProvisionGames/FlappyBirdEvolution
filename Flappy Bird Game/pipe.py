#!/usr/bin/env python

import pygame
import random
from defs import *  # import properties from defs.py


class Pipe:
    """ Object to store the information of pipe movements
    """

    def __init__(self, gameDisplay, x, y, pipe_type):
        """ Initialize the pipe movement class

        :param gameDisplay: pygame window size
        :param x: integer, x-position of the pipe
        :param y: integer, y-positiion of the pipe
        :param pipe_type: integer, 0 or 1 referring to the lower or upper pipe respectively
        """

        # Set the display width and height
        self.gameDisplay = gameDisplay

        # Set the pipe status to moving
        self.state = PIPE_MOVING

        # Set the pipe type of upper or lower
        self.pipe_type = pipe_type

        # Get the pipe image
        self.img = pygame.image.load(PIPE_FILENAME)

        # Set the location of the pipe image
        self.rect = self.img.get_rect()

        # Set the y-position of the top of the upper pipe
        if pipe_type == PIPE_UPPER:
            y = y - self.rect.height

        # Set the position of the pipe
        self.set_position(x, y)

    def set_position(self, x, y):
        """ Function to get the initially position of the pipe

        :param x: integer, x-position of the pipe
        :param y: integer, y-positiion of the pipe
        """

        # Set the x-position of the pipe to the left of the image
        self.rect.left = x

        # Set the y-position of the pipe image
        self.rect.top = y

    def move_position(self, dx, dy):
        """ Function to get new positions of the pipes when moving

        :param dx: integer, number to move the pipe in horizontal direction (x-position)
        :param dy: integer, number to move the pipe in vertical direction (y-position)
        """

        # Update the x-position of the pipe image
        self.rect.centerx += dx

        # Update the y-position of the pipe image
        self.rect.centery += dy

    def draw(self):
        """ Function to visualize the pipes
        """

        # Set the pipe image at the defined position
        self.gameDisplay.blit(self.img, self.rect)

    def check_status(self):
        """" Function to check the status of the pipe in movement
        """

        # Check if the pipe has left the screen
        if self.rect.right < 0:
            # Update the pipe status from moving to done
            self.state = PIPE_DONE

    def update(self, dt):
        """ Function to update the pipe movement variables

        :param dt: integer, time movement in milliseconds
        """

        # Check if the pipe is moving
        if self.state == PIPE_MOVING:
            # Get the new pipe position
            self.move_position(-(PIPE_SPEED * dt), 0)
            # Visualize the new pipe position
            self.draw()
            # Get the pipe status of moving or done moving
            self.check_status()


class PipeCollection:
    """ Object to store information of new pipes
    """

    def __init__(self, gameDisplay):
        """ Initialize the pipe collection class

        :param gameDisplay: pygame window size
        """

        # Set the display width and height
        self.gameDisplay = gameDisplay

        # Store created pipes
        self.pipes = []

    def add_new_pipe_pair(self, x):
        """ Function to add a new pair (upper and lower) of pipes

        :param x: integer, width of the game display
        """

        # Choose a random position between the boundaries and define the y-position for the upper pipe
        top_y = random.randint(PIPE_MIN, PIPE_MAX - PIPE_GAP_SIZE)
        # Define teh y-position of the lower pipe
        bottom_y = top_y + PIPE_GAP_SIZE

        # Tie the properties of the pipes together
        p1 = Pipe(self.gameDisplay, x, top_y, PIPE_UPPER)
        p2 = Pipe(self.gameDisplay, x, bottom_y, PIPE_LOWER)

        # Create the new pipes
        self.pipes.append(p1)
        self.pipes.append(p2)

    def create_new_set(self):
        """ Function to fill every new game with pipe pairs
        """

        # Clear the pipe list
        self.pipes = []
        # Define the x-position of the first pipe
        placed = PIPE_FIRST

        # As long as there is room left on the right of the display to place pipes
        while placed < DISPLAY_W:
            # Add a new pipe pair
            self.add_new_pipe_pair(placed)
            # Update x-position for a potential new pipe
            placed += PIPE_ADD_GAP

    def update(self, dt):
        """ Function to update the drawn pipes

        :param dt: integer, time movement in milliseconds
        """

        # Define th x-position of the pipe
        rightmost = 0

        # For every pipe in the list
        for p in self.pipes:
            # Update the drawn position of the pipe
            p.update(dt)

            # Only check upper pipes
            if p.pipe_type == PIPE_UPPER:
                # Look for the most right pipe
                if p.rect.left > rightmost:
                    # The position of the pipe is the left side of the image
                    rightmost = p.rect.left

        # If the most right pair of pipes had enough room to its right
        if rightmost < (DISPLAY_W - PIPE_ADD_GAP):
            # Spawn new pipes
            self.add_new_pipe_pair(DISPLAY_W)

        # Remove pipe pairs from the list that are not moving
        self.pipes = [p for p in self.pipes if p.state == PIPE_MOVING]
