import torch
import random
import numpy as numpy
from collections import deque
from snake_ai import BLOCK_SIZE, SnakeGameAI, Direction, Point


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class AI:

    def __init__(self):
        self.number_of_games = 0
        self.epsilon = 0 # controls randomness
        self.gamma = 0 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = None # TODO
        self.trainer = None # TODO


    def get_state(self, game):
        head = game.snake[0]
        point_left = Point(head.x - BLOCK_SIZE, head.y)
        point_right = Point(head.x + BLOCK_SIZE, head.y)
        point_up = Point(head.x, head.y - BLOCK_SIZE)
        point_down = Point(head.x, head.y + BLOCK_SIZE)
        
        direction_left = game.direction == Direction.LEFT
        direction_right = game.direction == Direction.RIGHT
        direction_up = game.direction == Direction.UP
        direction_down = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (direction_right and game.is_collision(point_right)) or 
            (direction_left and game.is_collision(point_left)) or 
            (direction_up and game.is_collision(point_up)) or 
            (direction_down and game.is_collision(point_down)),

            # Danger right
            (direction_up and game.is_collision(point_right)) or 
            (direction_down and game.is_collision(point_left)) or 
            (direction_left and game.is_collision(point_up)) or 
            (direction_right and game.is_collision(point_down)),

            # Danger left
            (direction_down and game.is_collision(point_right)) or 
            (direction_up and game.is_collision(point_left)) or 
            (direction_right and game.is_collision(point_up)) or 
            (direction_left and game.is_collision(point_down)),
            
            # Move direction
            direction_left,
            direction_right,
            direction_up,
            direction_down,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return numpy.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft() if Max_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step( state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves : tradeoff (between exploration vs exploitation)
        # more games -> smaller epsilon -> frequency of random moves decreases
        self.epsilon = 80 - self.number_of_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0,2) # 0 , 1 , 2
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model.predict(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    ai = AI()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = ai.get_state(game)

        # get move (based on current state)
        final_move = ai.get_action(state_old)

        #perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = ai.get_State(game)

        # train short memory
        ai.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        ai.remember(state_old, final_move, reward, state_new, done)

        # check if game over

        if done:
            # train long memory (replay memory / experience memory) & plot results
            game.reset()
            ai.number_of_games += 1
            ai.train_long_memory()

            if score > record: 
                record = score
                ai.model.save()

            print('Game: ',ai.number_of_games, 'Score: ', 'Record: ', record )

            # TODO: plot

if __name__ == '__main__':
    train()