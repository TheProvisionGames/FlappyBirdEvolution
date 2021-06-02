DISPLAY_W = 960
DISPLAY_H = 540
FPS = 30

DATA_FONT_SIZE = 18
DATA_FONT_COLOR = (40,40,40)
BG_FILENAME = '../BG.png'  #This is the positon of the background image
#If you are in a folder below with defs.py, the position is '../BG.png'

PIPE_FILENAME = '../Pipe.png'
PIPE_SPEED = 70/1000
PIPE_DONE = 1 #State number 1 means that the pipe is done moving
PIPE_MOVING = 0 #State number 0 means that the pipe is moving
PIPE_UPPER = 1 #State number 1 means that the pipe is at the top of the screen
PIPE_LOWER = 0 #State number 0 means that the pipe is at the bottom of the screen

PIPE_ADD_GAP = 160 #The gap between pairs of pipes
PIPE_MIN = 80 #How high the pipes may go (We are measuring distance from the top)
PIPE_MAX = 500 #How low the pipes may go
PIPE_START_X = DISPLAY_W #Pipes start on the right of the screen
PIPE_GAP_SIZE = 160 #Gap between two pipes in a pair (Where the bird can fly through)
PIPE_FIRST = 400 #The distance from the left of the screen to the first pipe pair

BIRD_FILENAME = '../Robin.png'
BIRD_START_SPEED = -0.32
BIRD_START_X = 200
BIRD_START_Y = 200
BIRD_ALIVE = 1
BIRD_DEAD = 0
GRAVITY = 0.001

#How many birds are spawned each generation
GENERATION_SIZE = 60

#Number of input, hidden and output nodes
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