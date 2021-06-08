# Import the pygame package
import pygame

# Import classes and properties of other Flappy Bird files
from defs import *
from pipe import PipeCollection
from bird import Bird


def update_label(data, title, font, x, y, gameDisplay):
    """ Function to set a text label at the game window

    :param data: integer, specific number belonging to the label
    :param title: string, text label to set
    :param font: integer, the size of the text
    :param x: integer, x-position of a text label
    :param y: integer, y-position of a text label
    :param gameDisplay: pygame window size
    :return:
        y: integer, y-position of a text label
    """

    # Set the format of the text label
    label = font.render('{} {}'.format(title, data), 1, DATA_FONT_COLOR)

    # Show the text label on the screen
    gameDisplay.blit(label, (x, y))

    # Return the y-position of the label
    return y


def update_data_labels(gameDisplay, dt, game_time, num_iterations, num_alive, font):
    """ Function to visualize game information while running the game

       :param gameDisplay: pygame window size
       :param dt: integer, time movement in milliseconds
       :param game_time: float, time in milliseconds the game is running
       :param num_iterations: integer, number of iterated games
       :param num_alive: integer, number of living birds
       :param font: integer, the size of the text
       """

    # Define the position of displaying the game information
    y_pos = 10
    x_pos = 10
    gap = 20

    # Show the number of frames per second
    y_pos = update_label(round(1000 / dt, 2), 'FPS', font, x_pos, y_pos + gap, gameDisplay)

    # Show the game time
    y_pos = update_label(round(game_time / 1000, 2), 'Game time', font, x_pos, y_pos + gap, gameDisplay)

    # Show the number of iterated games
    y_pos = update_label(num_iterations, 'Iterations', font, x_pos, y_pos + gap, gameDisplay)

    # Show number of living birds
    y_pos = update_label(num_alive, 'Alive', font, x_pos, y_pos + gap, gameDisplay)


def run_game():
    """ Function to start en keep running the Flappy Bird game
    """

    # Initialize the pygame
    pygame.init()

    # Set boundries for the game window
    gameDisplay = pygame.display.set_mode((DISPLAY_W, DISPLAY_H))

    # Set the title of the game
    pygame.display.set_caption('Learn to fly')

    # Set running to true to use the running loop
    running = True

    # Load the image of the background
    bgImg = pygame.image.load(BG_FILENAME)

    # Initialize the pipe collection class
    pipes = PipeCollection(gameDisplay)

    # Create a new set of pipes
    pipes.create_new_set()

    # Initialize the bird class
    bird = Bird(gameDisplay)

    # Set a chosen font and size
    label_font = pygame.font.SysFont("monospace", DATA_FONT_SIZE)

    # Get the running time information by using a built-in pygame clock
    clock = pygame.time.Clock()

    # Start with a delta and running time of 0
    dt = 0
    game_time = 0

    # Start with the first game iteration
    num_iterations = 1

    # From here the running game loop starts (all set values of above in this function will not play again)
    # Keep running the game while true
    while running:

        # Get the difference in time that takes the frames per second into account
        dt = clock.tick(FPS)

        # Add the delta time to total time
        game_time += dt

        # Draw background image from the upper right corner
        gameDisplay.blit(bgImg, (0, 0))

        # Loop over the pygame events
        for event in pygame.event.get():

            # Stop running the game if an event is quited
            if event.type == pygame.QUIT:
                running = False

            # Stop running the game when a key (with exception of the spcae bar) is pressed
            elif event.type == pygame.KEYDOWN:
                # Flap the bird when the key is the space bar
                if event.key == pygame.K_SPACE:
                    bird.jump()
                # Every other key will close the game
                else:
                    running = False

        # Update the drawn pipe
        pipes.update(dt)

        # Get the number of living birds
        num_alive = birds.update(dt, pipes.pipes)

        # Check if there are any birds left
        if num_alive == 0:
            # Create a new set of pipes
            pipes.create_new_set()

            # Restart the game time
            game_time = 0

            # Create a new set of bird population
            birds.evolve_population()

            # Update the number of iterated games
            num_iterations += 1

        # Update the text label information while running the game
        update_data_labels(gameDisplay, dt, game_time, num_iterations, num_alive, label_font)

        # Draw all of what is called on top of the layers that are there (30 times per second, FPS)
        # This means that python only draws over the old images and does not delete them
        pygame.display.update()


if __name__ == "__main__":
    # Call the play function only when running Python from this main.py file
    run_game()
