import os
import random
import numpy as np
from collections import deque
import tensorflow as tf
import pynput
import keyboard
from PIL import ImageGrab
from tensorflow.keras import activations
import cv2
from PIL import Image
from random import randrange


state_size = 400*300
action_size = 25
batch_size = 32
n_episodes = 1000
output_dir = 'model_output/MicroAI'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



mouse=pynput.mouse.Controller()


class Game:
    def reset(self):
        mouse.position = (0,0)
        print('1')
        while not (np.array(ImageGrab.grab(bbox=(350,300,350+100,300+100))) == np.array(Image.open('Images/Play_again.png'))).all():
            print('waiting for Play_again screen')

        mouse.position = (500,570)
        mouse.press(pynput.mouse.Button.left)
        cv2.waitKey(100)
        mouse.release(pynput.mouse.Button.left)
        cv2.waitKey(100)
        print('Play again')
        i = 0
        mouse.position = (650,300)
        while (np.array(ImageGrab.grab(bbox=(600,285,600+50,285+10))) != np.array(Image.open('Images/Select_map.png'))).all():
            print('waiting: '+ str(i))
            i+=1
        cv2.waitKey(1000)
        print('Select map')
        mouse.press(pynput.mouse.Button.left)
        mouse.release(pynput.mouse.Button.left)
        mouse.position = (500,150)
        mouse.press(pynput.mouse.Button.left)
        mouse.release(pynput.mouse.Button.left)
        print('H')
        while not (np.array(ImageGrab.grab(bbox=(230,150,230+100,150+10))) == np.array(Image.open('Images/Humvee_training.png'))).all():
            keyboard.press_and_release('h')
            print('Waiting for map')
            cv2.waitKey(100)

        #cv2.imshow('window',np.array(ImageGrab.grab(bbox=(230,175,230+75,175+25))))
        #cv2.imshow('window',np.array(Image.open('Images/Humvee_training.png')))
        
        print('Enter')
        keyboard.press_and_release('enter')
        mouse.position = (185,525)
        mouse.press(pynput.mouse.Button.left)
        cv2.waitKey(10)
        mouse.release(pynput.mouse.Button.left)
        print('Start Game')
        while not (np.array(ImageGrab.grab(bbox=(300,550,300+50,550+50))) == np.array(Image.open('Images/Game_started.png'))).all():
            print('Waiting to load')

        image = np.array(ImageGrab.grab(bbox=(0,0,800,600)))
        resized = cv2.resize(image, (400,300), interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = gray.reshape(1,400,300,)


        return gray
        #cv2.waitKey(10000)

    def save_screenshot(self):
        screenshot_pil = ImageGrab.grab(bbox=(300,200,300+25,200+25))
        screenshot_pil.save('Images/Defeat.png')
        
    def step(self, action):
        actions = []
        for x in range(5):
            for y in range(5):
                actions.append((800/7*(x+1),600/7*(y+1)))

        keyboard.press_and_release('q')
        mouse.position = (actions[action][0],actions[action][1])
        mouse.press(pynput.mouse.Button.right)
        cv2.waitKey(10)
        mouse.release(pynput.mouse.Button.right)

        image = np.array(ImageGrab.grab(bbox=(0,25,800,625)))
        resized = cv2.resize(image, (400,300), interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = gray.reshape(1,400,300)
        print ('shape: ' + str(np.shape(gray)))

        done = False
        reward = 10

        if  (np.array(ImageGrab.grab(bbox=(300,200,300+25,200+25))) == np.array(Image.open('Images/Defeat.png'))).all():
            done = True
            reward-=100

        return gray, reward, done

    

class QLearner:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 0.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        self.learning_rate = 0.001

        #self.model = tf.keras.models.load_model(output_dir+'/MicroAIweights_0000.hdf5')
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(400,300)))
        model.add(tf.keras.layers.Dense(1024, activation='relu'))
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(25, activation='relu'))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='categorical_crossentropy')

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        num = 0
        if np.random.rand() <= self.epsilon:
            num = randrange(25)
            print("random: " + str(num))
            return num
        pred = self.model.predict(state)[0]
        num = np.argmax(pred)
        print('pred: ' +str(pred))
        print('pred shape: ' +str(np.shape(pred)))
        return num

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward+self.gamma*np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose = 0)

        if (self.epsilon > self.epsilon_min):
            print('eplison: ' + str(self.epsilon))
            self.epsilon=self.epsilon*self.epsilon_decay

    def load(self, name):
        self.model.load(name)
    def save(self, name):
        self.model.save(name)


Sigma = QLearner(state_size, 25)
env = Game()
for layer in Sigma.model.layers:
    print(layer.output_shape)

for e in range(n_episodes):     
    state = env.reset()
    print('resetting: ' + str(e))
    for time in range(5000):
        action = Sigma.act(state)
        print(Sigma.act(state))
        next_state, reward, done = env.step(action)

        Sigma.remember(state,action,reward,next_state,done)
        state = next_state

        if (done):
            print('done')
            break
    if len(Sigma.memory) > batch_size:
        Sigma.replay(batch_size)
        print("replay")

    if e%50==0:
        Sigma.save(output_dir+'/episode'+str(e)+'.hdf5')


