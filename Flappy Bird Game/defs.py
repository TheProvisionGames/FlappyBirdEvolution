# Define the size of the game display
DISPLAY_W = 960
DISPLAY_H = 540

# Define the number of frames per second
FPS = 30

# Define the font size and color
DATA_FONT_SIZE = 18
DATA_FONT_COLOR = (40, 40, 40)

# Get the folder location of the background image
BG_FILENAME = '../BG.png'

# Get the folder location of the pipe image
PIPE_FILENAME = '../Pipe.png'
# Define the moving speed of the pipe images
PIPE_SPEED = 70/1000
# Define the state of the pipe when the pipe is done moving
PIPE_DONE = 1
# Define the state of the pipe when the pipe is moving
PIPE_MOVING = 0
# Define the upper pipe, which is the pipe is at the top of the screen
PIPE_UPPER = 1
# Define the lower pipe, which is the pipe is at the bottom of the screen
PIPE_LOWER = 0
# Define the gap between pairs of pipes
PIPE_ADD_GAP = 160
# Define how high the pipes may go (with measuring distance from the top)
PIPE_MIN = 80
# Define how low the pipes may go
PIPE_MAX = 500
# Set the starting postion of the pipes, which is on the right of the screen
PIPE_START_X = DISPLAY_W
# Define the gap between two pipes in a pair (Where the bird can fly through)
PIPE_GAP_SIZE = 160
# Define The distance from the left of the screen to the first pipe pair
PIPE_FIRST = 400

# Get the folder location of the bird image
BIRD_FILENAME = '../Robin.png'
# Define the starting speed of the bird
BIRD_START_SPEED = -0.32
# Define the start x-position of the bird
BIRD_START_X = 200
# Define the start y-position of the bird
BIRD_START_Y = 200
# Define the state of the bird when living
BIRD_ALIVE = 1
# Define the state of the bird when dead
BIRD_DEAD = 0
# Define the gravity of the bird when going down if it is not flapping
GRAVITY = 0.001

# Define how many birds are spawned each generation
GENERATION_SIZE = 60

# Define the number of input, hidden and output nodes
NNET_INPUTS = 2
NNET_HIDDEN = 5
NNET_OUTPUTS = 1
# Set the 'flap value'; flap when the value is higher or equal to 0.5
JUMP_CHANCE = 0.5

# Define the highest and lowest y-position for the bird to fly through the pipes
MAX_Y_DIFF = DISPLAY_H - PIPE_MIN - PIPE_GAP_SIZE / 2
MIN_Y_DIFF = PIPE_GAP_SIZE / 2 - PIPE_MAX

# Transform the lowest y-position to an absolute number
Y_SHIFT = abs(MIN_Y_DIFF)

# Define the maximal shift in distance
NORMALIZER = abs(MIN_Y_DIFF) + MAX_Y_DIFF

# Define the percentage of chance that the weights of a bad bird will be changed next generation
MUTATION_WEIGHT_MODIFY_CHANCE = 0.2
# Define the mix percentage, meaning the percentage of each of the two parent birds taken to create a new bird
MUTATION_ARRAY_MIX_PERC = 0.5
# Define the percentage of birds that will be deleted
MUTATION_CUT_OFF = 0.4
# Define the percentage of birds that are kept on the bad side
MUTATION_BAD_TO_KEEP = 0.2
# Define the probability that birds that are created from two parent birds have their weights changed
MUTATION_MODIFY_CHANCE_LIMIT = 0.4