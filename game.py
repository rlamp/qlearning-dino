import time

import cv2
import numpy as np
from PIL import ImageGrab
from selenium import webdriver
from selenium.webdriver.common.keys import Keys


class Game:
    # scaled down to 20 pixels + jump state
    observation_space_n = 2**21
    # Do nothing, jump
    action_space_n = 2

    jump_num = 2**20

    def __init__(self, disable_acceleration: bool = True) -> None:

        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--mute-audio")
        self._driver = webdriver.Chrome(options=chrome_options)

        self._driver.set_window_position(-10, 0)
        self._driver.set_window_size(670, 240)

        self._driver.get('chrome://dino/')
        time.sleep(1)

        if disable_acceleration:
            self._driver.execute_script("Runner.config.ACCELERATION=0")

        # start game, let animation play, and wait to crash
        self.jump()
        while not self.is_game_over():
            time.sleep(0.5)

        # game_over_state = cv2.imread('game_over.png', cv2.IMREAD_GRAYSCALE)
        # game_over_state = np.array(game_over_state, dtype=np.bool)

    @staticmethod
    def _scale_down(state: np.array, new_len: int = 20) -> np.array:
        # assert len(state) == 512
        step_size = len(state) // new_len
        res = np.zeros(new_len)

        for i in range(new_len):
            a = i * step_size
            b = a + step_size
            if state[a:b].any():
                res[i] = 1

        return res

    @staticmethod
    def _detect_obstacles(gray: np.array) -> np.array:
        _, ix = gray.shape
        res = [0] * ix
        res = np.array(res, dtype=np.uint8)
        for x in range(ix):
            column = gray[:, x]
            if not column.all():
                res[x] = 255

        return res

    @staticmethod
    def _detect_obstacles_scale_down(gray: np.array, new_len: int = 20) -> np.array:
        _, ix = gray.shape
        step_size = ix // new_len
        res = np.zeros(new_len, dtype=np.uint8)
        for i in range(new_len):
            a = i * step_size
            b = a + step_size
            if not gray[:, a:b].all():
                res[i] = 1

        return res

    def get_state(self) -> int:
        bbox_x, bbox_y = 112, 111
        bbox_width, bbox_height = 492, 108
        image = np.array(ImageGrab.grab(
            bbox=(bbox_x, bbox_y, bbox_x+bbox_width, bbox_y+bbox_height)))

        # cv2.imshow('window', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]

        if np.median(gray[0, :]) == 0:
            gray = np.invert(gray)

        # state = self._detect_obstacles(gray)
        # state = self._scale_down(state)
        state = self._detect_obstacles_scale_down(gray)
        state = state.tolist()
        # Append additional info: is jumping
        state.append(int(self.is_jumping()))

        # Convert binary list to in int
        state_num = int(''.join(str(int(x)) for x in state), 2)
        return state_num

    def take_action(self, action: int) -> tuple:
        if action == 1:
            self.jump()

        state = self.get_state()
        game_over = self.is_game_over()

        if game_over:
            reward = -1000
        # Is jumping
        elif (state % 2 == 1):
            # Reward for jumping over stuff
            if state >= self.jump_num:
                reward = 5
            # Penalty for just jumping
            else:
                reward = -1
        # Reward for living/staying on ground
        else:
            reward = 1

        return state, reward, game_over

    def jump(self) -> None:
        self._driver.find_element_by_tag_name('body').send_keys(Keys.SPACE)

    # def is_jumping(self, gray: np.array) -> bool:
    #     pass
    def is_jumping(self) -> bool:
        return self._driver.execute_script("return Runner.instance_.tRex.jumping")

    # def is_game_over(self, state: np.array) -> bool:
    #     state = np.array(state, dtype=np.bool)
    #     check_state = game_over_state * state
    #     return np.all(check_state == game_over_state)
    def is_game_over(self) -> bool:
        return self._driver.execute_script('return Runner.instance_.crashed')

    def get_score(self) -> None:
        score_array = self._driver.execute_script(
            "return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array)
        return int(score)

    def reset(self) -> list:
        # self._driver.find_element_by_tag_name('body').send_keys(Keys.ENTER)
        self._driver.execute_script('Runner.instance_.restart()')
        time.sleep(.5)
        return self.get_state()

    def end(self) -> None:
        self._driver.quit()
